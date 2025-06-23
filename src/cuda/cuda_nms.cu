// cuda_nms_improved_fp16_uint16.cu
// Implements fused copy+threshold, GPU‐side filtering via atomics, CUB GPU sorting via DeviceRadixSort,
// suppression and feature gathering, all using fp16 and uint16_t for internal storage.

#include "cuda_nms.h"
#include "cuda_stuff.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>      // for std::isfinite
#include <cstdint>    // for uint16_t
#include <cuda_fp16.h>

//-------------------------------------------------------------------------------------------------
// Macro for CUDA error checking
//-------------------------------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                      \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                 \
            std::exit(1);                                                    \
        }                                                                     \
    } while (0)

//-------------------------------------------------------------------------------------------------
// 1) Fused copy+threshold+collect passing scores/indices via atomic
//-------------------------------------------------------------------------------------------------
__global__ static void filterAndCollectKernel(
    const float*      __restrict__ all_data,      // [ (4+numClasses) × numBoxes ]
    int                         numBoxes,
    int                         classIdx,
    float                       thr,
    int*            __restrict__ d_numCand,       // scalar per-class
    uint16_t*       __restrict__ d_filteredIdx,   // [numBoxes]
    __half*         __restrict__ d_filteredScores // [numBoxes]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoxes) return;
    float s = all_data[(4 + classIdx) * (size_t)numBoxes + tid];
    // defensive: scores and threshold must be finite
    // assert(std::isfinite(s) && std::isfinite(thr));
    if (s >= thr) {
        int pos = atomicAdd(d_numCand, 1);
        if (pos < numBoxes) {
            d_filteredIdx[pos]    = static_cast<uint16_t>(tid);
            d_filteredScores[pos] = __float2half_rn(s);
        }
    }
}

__global__ static void filterAndCollectKernelFP16(
    const __half*    __restrict__ all_data,      // [ (4+numClasses) × numBoxes ]
    int                         numBoxes,
    int                         classIdx,
    float                       thr,
    int*            __restrict__ d_numCand,       // scalar per-class
    uint16_t*       __restrict__ d_filteredIdx,   // [numBoxes]
    __half*         __restrict__ d_filteredScores // [numBoxes]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoxes) return;
    // load fp16, convert to float
    float s = __half2float(all_data[(4 + classIdx) * (size_t)numBoxes + tid]);
    if (s >= thr) {
        int pos = atomicAdd(d_numCand, 1);
        if (pos < numBoxes) {
            d_filteredIdx[pos]    = static_cast<uint16_t>(tid);
            d_filteredScores[pos] = __float2half_rn(s);
        }
    }
}

//-------------------------------------------------------------------------------------------------
// Kernel A: Convert (cx,cy,w,h) → (x1,y1,x2,y2), storing in fp16
//-------------------------------------------------------------------------------------------------
__global__ static void centerToCornerKernel(
    const float*  __restrict__ in_centers,  // [4 × numBoxes]
    __half*       __restrict__ out_corners, // [numBoxes × 4]
    int                     numBoxes ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBoxes) return;

    float cx = in_centers[0 * numBoxes + idx];
    float cy = in_centers[1 * numBoxes + idx];
    float w  = in_centers[2 * numBoxes + idx];
    float h  = in_centers[3 * numBoxes + idx];

    float x1 = cx - 0.5f * w;
    float y1 = cy - 0.5f * h;
    float x2 = cx + 0.5f * w;
    float y2 = cy + 0.5f * h;

    out_corners[idx * 4 + 0] = __float2half_rn(x1);
    out_corners[idx * 4 + 1] = __float2half_rn(y1);
    out_corners[idx * 4 + 2] = __float2half_rn(x2);
    out_corners[idx * 4 + 3] = __float2half_rn(y2);
}

__global__ static void centerToCornerKernelFP16(
    const __half*  __restrict__ in_centers,  // [4 × numBoxes] in fp16
    __half*        __restrict__ out_corners, // [numBoxes × 4]
    int                      numBoxes ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes) return;

    float cx = __half2float(in_centers[0 * numBoxes + idx]);
    float cy = __half2float(in_centers[1 * numBoxes + idx]);
    float w  = __half2float(in_centers[2 * numBoxes + idx]);
    float h  = __half2float(in_centers[3 * numBoxes + idx]);

    float x1 = cx - 0.5f * w;
    float y1 = cy - 0.5f * h;
    float x2 = cx + 0.5f * w;
    float y2 = cy + 0.5f * h;

    out_corners[idx * 4 + 0] = __float2half_rn(x1);
    out_corners[idx * 4 + 1] = __float2half_rn(y1);
    out_corners[idx * 4 + 2] = __float2half_rn(x2);
    out_corners[idx * 4 + 3] = __float2half_rn(y2);
}

//-------------------------------------------------------------------------------------------------
// Kernel B: Build pairwise IoU mask in 64-box tiles (fixed stride handling)
//-------------------------------------------------------------------------------------------------
__global__ static void buildPairwiseMaskKernel(
    const __half*       __restrict__ corners,    // [numBoxes × 4]
    const uint16_t*     __restrict__ sortedIdx,  // [numBoxes]
    int                             numBoxes,
    float                           iouThreshold,
    int                             dynStride,    // ceil(numBoxes/64)
    int                             memStride,    // workspace->maxMaskStride
    unsigned long long* __restrict__ maskOut      // [numBoxes × memStride]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y;
    if (i >= numBoxes || w >= dynStride) return;

    int idx_i = sortedIdx[i];
    float x1_i = __half2float(corners[idx_i * 4 + 0]);
    float y1_i = __half2float(corners[idx_i * 4 + 1]);
    float x2_i = __half2float(corners[idx_i * 4 + 2]);
    float y2_i = __half2float(corners[idx_i * 4 + 3]);
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);

    unsigned long long mask = 0ULL;
    int startJ = w * 64;
    int endJ   = min(startJ + 64, numBoxes);
    for (int j = startJ; j < endJ; ++j) {
        if (j <= i) continue;
        int idx_j = sortedIdx[j];
        float x1_j = __half2float(corners[idx_j * 4 + 0]);
        float y1_j = __half2float(corners[idx_j * 4 + 1]);
        float x2_j = __half2float(corners[idx_j * 4 + 2]);
        float y2_j = __half2float(corners[idx_j * 4 + 3]);
        float area_j = (x2_j - x1_j) * (y2_j - y1_j);

        float xx1 = fmaxf(x1_i, x1_j);
        float yy1 = fmaxf(y1_i, y1_j);
        float xx2 = fminf(x2_i, x2_j);
        float yy2 = fminf(y2_i, y2_j);
        float w_int = max(0.0f, xx2 - xx1);
        float h_int = max(0.0f, yy2 - yy1);
        float inter = w_int * h_int;
        float ovr = inter / (area_i + area_j - inter);
        if (ovr > iouThreshold) {
            mask |= (1ULL << (j - startJ));
        }
    }
    maskOut[i * memStride + w] = mask;
}

//-------------------------------------------------------------------------------------------------
// Kernel C: Suppression per class (serial on one thread)
//-------------------------------------------------------------------------------------------------
__global__ static void doSuppressionKernel(
    const uint16_t*         __restrict__ sortedIdx,   // [numBoxes]
    const unsigned long long* __restrict__ maskWords, // [numBoxes × memStride]
    int                             numBoxes,
    int                             memStride,
    int                             dynStride,
    int                             maxOut,
    int*            __restrict__    outKeepCount,      // scalar
    uint16_t*       __restrict__    outKeepIdx         // [maxOut]
) {
    extern __shared__ unsigned long long s_globalSuppress[];
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int w = 0; w < dynStride; ++w) {
        s_globalSuppress[w] = 0ULL;
    }
    int keepCnt = 0;

    for (int i = 0; i < numBoxes; ++i) {
        int wordIdx = i >> 6, bitIdx = i & 63;
        if ((s_globalSuppress[wordIdx] & (1ULL << bitIdx)) == 0ULL) {
            if (keepCnt < maxOut) {
                outKeepIdx[keepCnt] = sortedIdx[i];
            }
            keepCnt++;
            for (int w = wordIdx; w < dynStride; ++w) {
                s_globalSuppress[w] |= maskWords[i * memStride + w];
            }
        }
    }
    *outKeepCount = (keepCnt < maxOut ? keepCnt : maxOut);
}

//-------------------------------------------------------------------------------------------------
// Kernel D: Gather features for kept indices (with bounds asserts)
//-------------------------------------------------------------------------------------------------
__global__ static void gatherFeaturesKernel(
    const float*      __restrict__ src,        // [rowSize × numBoxes]
    const uint16_t*   __restrict__ keptIdx,    // [totalKept]
    int                         numBoxes,
    int                         rowSize,
    int                         totalKept,
    float*            __restrict__ dst         // [totalKept × rowSize]
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= totalKept || r >= rowSize) return;
    int box = keptIdx[k];
    assert(box >= 0 && box < numBoxes);
    dst[k * rowSize + r] = src[r * numBoxes + box];
}

__global__ static void gatherFeaturesKernelFP16(
    const __half*       __restrict__ src,        // [rowSize × numBoxes] in fp16
    const uint16_t*     __restrict__ keptIdx,    // [totalKept]
    int                             numBoxes,
    int                             rowSize,
    int                             totalKept,
    float*          __restrict__    dst         // [totalKept × rowSize], always float
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= totalKept || r >= rowSize) return;
    int box = keptIdx[k];
    float v = __half2float(src[r * numBoxes + box]);
    dst[k * rowSize + r] = v;
}

//-------------------------------------------------------------------------------------------------
// Workspace struct with new buffers for GPU filtering/sorting
//-------------------------------------------------------------------------------------------------
struct CudaNMSWorkspace_t {
    int             maxBoxes;
    int             maxClasses;
    int             maxOutPerClass;
    int             maxMaskStride;   // ceil(maxBoxes/64)
    cudaStream_t    stream;

    __half*         d_corners;        // [maxBoxes × 4]
    uint16_t*       d_sortedIdx;      // [maxBoxes × maxClasses]

    int*            d_numCand;        // scalar per-class
    uint16_t*       d_filteredIdx;    // [maxBoxes]
    __half*         d_filteredScores; // [maxBoxes]
    void*           d_cubTemp;
    size_t          cubTempBytes;

    unsigned long long* d_mask;       // [maxBoxes × maxMaskStride × maxClasses]
    int*            d_keepCount;      // [maxClasses]
    uint16_t*       d_keptIdx;        // [maxClasses × maxOutPerClass]
};

//-------------------------------------------------------------------------------------------------
// Allocate a new workspace and return its handle.
//-------------------------------------------------------------------------------------------------
CudaNMSHandle cuda_nms_allocate_workspace(
    int           maxBoxes,
    int           maxClasses,
    int           maxOutPerClass,
    cudaStream_t  stream
) {
    assert(maxBoxes>0 && maxClasses>0 && maxOutPerClass>0);
    auto* ws = (CudaNMSWorkspace_t*)std::malloc(sizeof(CudaNMSWorkspace_t));
    ws->maxBoxes       = maxBoxes;
    ws->maxClasses     = maxClasses;
    ws->maxOutPerClass = maxOutPerClass;
    ws->stream         = stream;
    ws->maxMaskStride  = (maxBoxes + 63) / 64;

    ws->d_corners=(__half *)cuda_malloc(sizeof(__half)    * maxBoxes * 4);
    ws->d_sortedIdx=(uint16_t *)cuda_malloc(sizeof(uint16_t) * maxBoxes * maxClasses);
    ws->d_mask=(unsigned long long *)cuda_malloc(sizeof(unsigned long long) * maxBoxes * ws->maxMaskStride * maxClasses);
    ws->d_keepCount=(int *)cuda_malloc(sizeof(int)       * maxClasses);
    ws->d_keptIdx=(uint16_t *)cuda_malloc(sizeof(uint16_t) * maxClasses * maxOutPerClass);
    ws->d_numCand=(int *)cuda_malloc(sizeof(int));
    ws->d_filteredIdx=(uint16_t *)cuda_malloc(sizeof(uint16_t) * maxBoxes);
    ws->d_filteredScores=(__half *)cuda_malloc(sizeof(__half)    * maxBoxes);

    // Get required CUB temp size for fp16-key, uint16_t-value sort:
    ws->cubTempBytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, ws->cubTempBytes,
        ws->d_filteredScores, ws->d_filteredScores,
        ws->d_filteredIdx,   ws->d_filteredIdx,
        maxBoxes,
        0, sizeof(__half)*8,
        stream);
    ws->d_cubTemp=cuda_malloc(ws->cubTempBytes);

    return (CudaNMSHandle)ws;
}

//-------------------------------------------------------------------------------------------------
// Free all GPU buffers and the struct itself.
//-------------------------------------------------------------------------------------------------
void cuda_nms_free_workspace(CudaNMSHandle handle) {
    if (!handle) return;
    auto* ws = (CudaNMSWorkspace_t*)handle;
    cuda_free(ws->d_corners);
    cuda_free(ws->d_sortedIdx);
    cuda_free(ws->d_mask);
    cuda_free(ws->d_keepCount);
    cuda_free(ws->d_keptIdx);
    cuda_free(ws->d_numCand);
    cuda_free(ws->d_filteredIdx);
    cuda_free(ws->d_filteredScores);
    cuda_free(ws->d_cubTemp);
    std::free(ws);
}

//-------------------------------------------------------------------------------------------------
// cuda_nms_run: uses fused filter+collect, CUB radix sort, then mask & suppression
//-------------------------------------------------------------------------------------------------
void cuda_nms_run(
    CudaNMSHandle                     handle,
    const float*                      all_data,       // [batch × rowSize × numBoxes]
    bool                              data_is_fp16,
    int                               batch_size,
    int                               numBoxes,
    int                               numClasses,
    int                               rowSize,
    float                             scoreThreshold,
    float                             iouThreshold,
    int                               maxOutPerClass,
    std::vector<std::vector<uint16_t>>& keptIndices,  // now uint16_t
    cudaStream_t                      stream
) {
    assert(handle);
    auto* ws = (CudaNMSWorkspace_t*)handle;
    assert(numBoxes>0 && numBoxes<=ws->maxBoxes);
    assert(numClasses>0 && numClasses<=ws->maxClasses);
    assert(maxOutPerClass>0 && maxOutPerClass<=ws->maxOutPerClass);
    assert(rowSize>=4+numClasses);
    assert(all_data);

    keptIndices.clear();
    keptIndices.resize(batch_size * numClasses);
    int h_numCand;

    for (int b = 0; b < batch_size; ++b) {
        const float* srcBatch = all_data + (size_t)b * rowSize * numBoxes;
        const float*  srcFloat = all_data  + (size_t)b * rowSize * numBoxes;
        const __half* srcHalf  = reinterpret_cast<const __half*>(all_data)
                                   + (size_t)b * rowSize * numBoxes;
        // 1) Convert centers → corners
        int t1 = 256, b1 = (numBoxes + t1 - 1) / t1;
        //centerToCornerKernel<<<b1, t1, 0, stream>>>(srcBatch, ws->d_corners, numBoxes);
        if (!data_is_fp16) {
            centerToCornerKernel<<<b1,t1,0,stream>>>(srcFloat, ws->d_corners, numBoxes);
        } else {
            centerToCornerKernelFP16<<<b1,t1,0,stream>>>(srcHalf, ws->d_corners, numBoxes);
        }
        CUDA_CHECK(cudaGetLastError());

        // 2) Per-class filtering, sorting, suppression
        for (int c = 0; c < numClasses; ++c) {
            CUDA_CHECK(cudaMemsetAsync(ws->d_numCand, 0, sizeof(int), stream));
            int t2 = 256, b2 = (numBoxes + t2 - 1) / t2;
            /*filterAndCollectKernel<<<b2, t2, 0, stream>>>(
                srcBatch, numBoxes, c, scoreThreshold,
                ws->d_numCand,
                ws->d_filteredIdx,
                ws->d_filteredScores);*/
            if (!data_is_fp16) {
                filterAndCollectKernel<<<b2,t2,0,stream>>>(
                    srcFloat, numBoxes, c, scoreThreshold,
                    ws->d_numCand,
                    ws->d_filteredIdx,
                    ws->d_filteredScores);
            } else {
                filterAndCollectKernelFP16<<<b2,t2,0,stream>>>(
                    srcHalf, numBoxes, c, scoreThreshold,
                    ws->d_numCand,
                    ws->d_filteredIdx,
                    ws->d_filteredScores);
            }
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(
                &h_numCand, ws->d_numCand, sizeof(int),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            h_numCand = std::min(h_numCand, ws->maxBoxes);
            if (h_numCand <= 0) continue;

            // …and now the CUB radix‐sort call with the correct signature:
            cub::DeviceRadixSort::SortPairsDescending(
                ws->d_cubTemp, ws->cubTempBytes,
                ws->d_filteredScores, ws->d_filteredScores,
                ws->d_filteredIdx,     ws->d_filteredIdx,
                h_numCand,
                0, sizeof(__half)*8,
                stream);

            // copy sorted indices into global arrays
            CUDA_CHECK(cudaMemcpyAsync(
                ws->d_sortedIdx + (size_t)c * ws->maxBoxes,
                ws->d_filteredIdx,
                sizeof(uint16_t) * h_numCand,
                cudaMemcpyDeviceToDevice,
                stream));

            int dynStride = (h_numCand + 63) / 64;
            int memStride = ws->maxMaskStride;
            dim3 threads3(32, 1), blocks3((h_numCand + 31) / 32, dynStride);
            buildPairwiseMaskKernel<<<blocks3, threads3, 0, stream>>>(
                ws->d_corners,
                ws->d_sortedIdx + c * ws->maxBoxes,
                h_numCand,
                iouThreshold,
                dynStride,
                memStride,
                ws->d_mask + (size_t)c * ws->maxBoxes * memStride);
            CUDA_CHECK(cudaGetLastError());

            int sharedBytes = sizeof(unsigned long long) * memStride;
            doSuppressionKernel<<<1, 1, sharedBytes, stream>>>(
                ws->d_sortedIdx + c * ws->maxBoxes,
                ws->d_mask + c * ws->maxBoxes * memStride,
                h_numCand,
                memStride,
                dynStride,
                maxOutPerClass,
                ws->d_keepCount + c,
                ws->d_keptIdx + (size_t)c * maxOutPerClass);
            CUDA_CHECK(cudaGetLastError());

            int hostKeepCnt;
            CUDA_CHECK(cudaMemcpyAsync(
                &hostKeepCnt,
                ws->d_keepCount + c,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            hostKeepCnt = std::min(hostKeepCnt, maxOutPerClass);

            if (hostKeepCnt > 0) {
                std::vector<uint16_t> tmp(hostKeepCnt);
                CUDA_CHECK(cudaMemcpyAsync(
                    tmp.data(),
                    ws->d_keptIdx + (size_t)c * maxOutPerClass,
                    sizeof(uint16_t) * hostKeepCnt,
                    cudaMemcpyDeviceToHost,
                    stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                keptIndices[b * numClasses + c] = std::move(tmp);
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

//-------------------------------------------------------------------------------------------------
// Gather kept outputs back to host (using gatherFeaturesKernel)
//-------------------------------------------------------------------------------------------------
void cuda_nms_gather_kept_outputs(
    const float*                          deviceOutputDev,   // [rowSize × numBoxes]
    bool                                  data_is_fp16,
    int                                   numBoxes,
    int                                   rowSize,
    const std::vector<std::vector<uint16_t>>& keptIndices,
    std::vector<float>&                   hostGathered
) {
    int totalClasses = static_cast<int>(keptIndices.size());
    int totalKept = 0;
    for (int c = 0; c < totalClasses; ++c) totalKept += static_cast<int>(keptIndices[c].size());
    if (totalKept == 0) { hostGathered.clear(); return; }

    std::vector<uint16_t> keptFlat;
    keptFlat.reserve(totalKept);
    for (auto &v : keptIndices) keptFlat.insert(keptFlat.end(), v.begin(), v.end());

    uint16_t* d_keptFlat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keptFlat, sizeof(uint16_t) * totalKept));
    CUDA_CHECK(cudaMemcpy(d_keptFlat, keptFlat.data(),
               sizeof(uint16_t) * totalKept, cudaMemcpyHostToDevice));

    float* d_gathered = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gathered, sizeof(float) * totalKept * rowSize));

    const int TILE_X = 32, TILE_Y = 32;
    dim3 blockDim(TILE_X, TILE_Y);
    dim3 gridDim((totalKept + TILE_X - 1) / TILE_X,
                 (rowSize   + TILE_Y - 1) / TILE_Y);
    /*gatherFeaturesKernel<<<gridDim, blockDim>>>(
        deviceOutputDev, d_keptFlat,
        numBoxes, rowSize, totalKept, d_gathered);*/
    const float*  srcFloat = deviceOutputDev;
    const __half* srcHalf  = reinterpret_cast<const __half*>(deviceOutputDev);
    if (!data_is_fp16) {
        gatherFeaturesKernel<<<gridDim, blockDim>>>(
            srcFloat, d_keptFlat,
            numBoxes, rowSize, totalKept, d_gathered);
    } else {
        gatherFeaturesKernelFP16<<<gridDim, blockDim>>>(
            srcHalf, d_keptFlat,
            numBoxes, rowSize, totalKept, d_gathered);
    }
    CUDA_CHECK(cudaGetLastError());

    hostGathered.resize((size_t)totalKept * rowSize);
    CUDA_CHECK(cudaMemcpy(hostGathered.data(), d_gathered,
               sizeof(float) * totalKept * rowSize,
               cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_keptFlat));
    CUDA_CHECK(cudaFree(d_gathered));
}
