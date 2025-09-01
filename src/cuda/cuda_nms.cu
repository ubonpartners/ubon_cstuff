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

// Tuneable; 256 is a safe default on most GPUs.
#ifndef FILTER_BLOCK_SIZE
#define FILTER_BLOCK_SIZE 256
#endif

// Float input version
__global__ void filterAndCollectKernel_BlockScan(
    const float*      __restrict__ all_data,      // [ (4+numClasses) × numBoxes ]
    int                               numBoxes,
    int                               classIdx,
    float                             thr,
    int*            __restrict__      d_numCand,       // scalar (reset to 0 per class)
    uint16_t*       __restrict__      d_filteredIdx,   // [numBoxes]
    __half*         __restrict__      d_filteredScores // [numBoxes]
) {
    using BlockScan = cub::BlockScan<int, FILTER_BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int block_base;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const bool in_range = (tid < numBoxes);

    // Load score
    float s = 0.0f;
    if (in_range) {
        s = all_data[(4 + classIdx) * (size_t)numBoxes + tid];
    }

    // Predicate: keep?
    int keep = (in_range && (s >= thr)) ? 1 : 0;

    // In-block exclusive prefix sum to get local offset; also get block total
    int local_off = 0;
    int block_count = 0;
    BlockScan(scan_storage).ExclusiveSum(keep, local_off, block_count);

    // One atomic per block to reserve a contiguous range
    if (threadIdx.x == 0) {
        block_base = (block_count > 0) ? atomicAdd(d_numCand, block_count) : 0;
    }
    __syncthreads();

    // Write kept items
    if (keep) {
        int pos = block_base + local_off;
        d_filteredIdx[pos]    = static_cast<uint16_t>(tid);
        d_filteredScores[pos] = __float2half_rn(s);
    }
}

// FP16 input version
__global__ void filterAndCollectKernelFP16_BlockScan(
    const __half*    __restrict__ all_data,       // [ (4+numClasses) × numBoxes ] fp16
    int                               numBoxes,
    int                               classIdx,
    float                             thr,
    int*            __restrict__      d_numCand,       // scalar (reset to 0 per class)
    uint16_t*       __restrict__      d_filteredIdx,   // [numBoxes]
    __half*         __restrict__      d_filteredScores // [numBoxes]
) {
    using BlockScan = cub::BlockScan<int, FILTER_BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int block_base;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const bool in_range = (tid < numBoxes);

    float s = 0.0f;
    if (in_range) {
        s = __half2float(all_data[(4 + classIdx) * (size_t)numBoxes + tid]);
    }

    int keep = (in_range && (s >= thr)) ? 1 : 0;

    int local_off = 0;
    int block_count = 0;
    BlockScan(scan_storage).ExclusiveSum(keep, local_off, block_count);

    if (threadIdx.x == 0) {
        block_base = (block_count > 0) ? atomicAdd(d_numCand, block_count) : 0;
    }
    __syncthreads();

    if (keep) {
        int pos = block_base + local_off;
        d_filteredIdx[pos]    = static_cast<uint16_t>(tid);
        d_filteredScores[pos] = __float2half_rn(s);
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
// Warp-cooperative mask build with early reject and warp reduction.
// Each warp handles one "i". Each lane handles a subset of j in the 64-wide tile.
__global__ static void buildPairwiseMaskKernel(
    const __half*       __restrict__ corners,    // [numBoxes × 4]
    const uint16_t*     __restrict__ sortedIdx,  // [numBoxes]
    int                             numBoxes,
    float                           iouThreshold,
    int                             dynStride,    // ceil(numBoxes/64)
    int                             memStride,    // workspace->maxMaskStride
    unsigned long long* __restrict__ maskOut      // [numBoxes × memStride]
) {
    const int WARP = 32;
    const unsigned int fullMask = __activemask();

    // One warp = one candidate i
    int warpId         = threadIdx.x >> 5;          // 0..(warpsPerBlock-1)
    int lane           = threadIdx.x & 31;          // 0..31
    int warpsPerBlock  = blockDim.x >> 5;

    int i = blockIdx.x * warpsPerBlock + warpId;
    int w = blockIdx.y;
    if (i >= numBoxes || w >= dynStride) return;

    // Load box i
    int idx_i = sortedIdx[i];
    float x1_i = __half2float(corners[idx_i * 4 + 0]);
    float y1_i = __half2float(corners[idx_i * 4 + 1]);
    float x2_i = __half2float(corners[idx_i * 4 + 2]);
    float y2_i = __half2float(corners[idx_i * 4 + 3]);
    float area_i = fmaxf(0.0f, x2_i - x1_i) * fmaxf(0.0f, y2_i - y1_i);

    // Tile range [startJ, endJ)
    const int startJ = w * 64;
    const int endJ   = min(startJ + 64, numBoxes);

    // Each lane accumulates bits for the j's it owns
    unsigned long long laneMask = 0ULL;

    // Process tile in 32-wide chunks so the warp can ballot
    for (int base = startJ; base < endJ; base += WARP) {
        int j = base + lane;
        // Active if in bounds and in strictly-lower score order (j > i)
        const bool active = (j < endJ) & (j > i);

        float x1_j, y1_j, x2_j, y2_j;
        bool overlap = false;

        if (active) {
            int idx_j = sortedIdx[j];
            x1_j = __half2float(corners[idx_j * 4 + 0]);
            y1_j = __half2float(corners[idx_j * 4 + 1]);
            x2_j = __half2float(corners[idx_j * 4 + 2]);
            y2_j = __half2float(corners[idx_j * 4 + 3]);

            // Fast AABB reject
            overlap = !(x2_j <= x1_i || x2_i <= x1_j || y2_j <= y1_i || y2_i <= y1_j);
        }

        // If no lanes in this 32-wide chunk overlap, skip the heavy IoU math
        unsigned int overlapMask = __ballot_sync(fullMask, active && overlap);
        if (overlapMask == 0u) {
            continue;
        }

        // Compute IoU only for lanes with potential overlap
        if (active && overlap) {
            float area_j = fmaxf(0.0f, x2_j - x1_j) * fmaxf(0.0f, y2_j - y1_j);

            float xx1 = fmaxf(x1_i, x1_j);
            float yy1 = fmaxf(y1_i, y1_j);
            float xx2 = fminf(x2_i, x2_j);
            float yy2 = fminf(y2_i, y2_j);
            float w_int = fmaxf(0.0f, xx2 - xx1);
            float h_int = fmaxf(0.0f, yy2 - yy1);
            float inter = w_int * h_int;

            float denom = (area_i + area_j - inter);
            if (denom > 0.0f) {
                float ovr = inter / denom;
                if (ovr > iouThreshold) {
                    int bit = j - startJ; // 0..63 within this tile
                    laneMask |= (1ULL << bit);
                }
            }
        }
    }

    // Warp-wide OR-reduction of the 64-bit laneMask (split into 2x32-bit halves)
    unsigned int lo = (unsigned int)(laneMask & 0xFFFFFFFFULL);
    unsigned int hi = (unsigned int)(laneMask >> 32);

    // Reduce within the warp
    for (int offs = 16; offs > 0; offs >>= 1) {
        lo |= __shfl_down_sync(fullMask, lo, offs);
        hi |= __shfl_down_sync(fullMask, hi, offs);
    }

    // Lane 0 writes the final 64-bit word
    if (lane == 0) {
        maskOut[(size_t)i * memStride + w] = ( (unsigned long long)hi << 32 ) | (unsigned long long)lo;
    }
}


__global__ static void doSuppressionKernel(
    const uint16_t*           __restrict__ sortedIdx,   // [numBoxes]
    const unsigned long long* __restrict__ maskWords,   // [numBoxes × memStride]
    int                       numBoxes,
    int                       memStride,
    int                       dynStride,
    int                       maxOut,
    int*                      __restrict__ outKeepCount, // scalar
    uint16_t*                 __restrict__ outKeepIdx    // [maxOut]
) {
    extern __shared__ unsigned long long s_globalSuppress[]; // length = dynStride
    const int T = blockDim.x;

    // Zero shared suppression bitmap
    for (int w = threadIdx.x; w < dynStride; w += T) {
        s_globalSuppress[w] = 0ULL;
    }
    __syncthreads();

    __shared__ int sh_keepCnt;
    __shared__ int sh_i;        // current candidate index
    __shared__ int sh_take;     // 0/1: we keep this candidate
    __shared__ int sh_done;     // 0/1: stop condition
    if (threadIdx.x == 0) {
        sh_keepCnt = 0;
        sh_i       = 0;
        sh_done    = 0;
    }
    __syncthreads();

    while (true) {
        // ---- Pick next unsuppressed candidate index (skip suppressed spans) ----
        if (threadIdx.x == 0) {
            sh_take = 0; // default
            if (sh_i >= numBoxes) {
                sh_done = 1;
            } else {
                int wordIdx = sh_i >> 6;
                int bitIdx  = sh_i & 63;

                // Bits available (not yet suppressed) from current bit to end of this word
                unsigned long long avail = ~s_globalSuppress[wordIdx] & (~0ULL << bitIdx);

                // If none in this word, jump to next word that has any availability
                while (avail == 0ULL) {
                    ++wordIdx;
                    sh_i = wordIdx << 6;
                    if (sh_i >= numBoxes) { sh_done = 1; break; }
                    avail = ~s_globalSuppress[wordIdx]; // full word from bit 0
                }

                if (!sh_done) {
                    // First available (least-significant) set bit = next unsuppressed candidate
                    const int nextBit = __ffsll(avail) - 1; // 0..63
                    sh_i = (wordIdx << 6) + nextBit;

                    // Decide to take (greedy) if we still have quota
                    if (sh_keepCnt < maxOut) {
                        outKeepIdx[sh_keepCnt] = sortedIdx[sh_i];
                        sh_take = 1;
                    }
                }
            }
        }
        __syncthreads();  // Barrier A: broadcast sh_i, sh_take, sh_done

        if (sh_done) break;

        // ---- If we kept this candidate, OR its mask row into the global suppression map ----
        if (sh_take) {
            // The mask for candidate `sh_i` starts at word index (sh_i >> 6)
            const int wordStart = sh_i >> 6;

            // Warp/block-cooperative OR across the row:
            // Each thread handles a distinct subset of 64-bit words to reduce loop time.
            for (int w = wordStart + threadIdx.x; w < dynStride; w += blockDim.x) {
                unsigned long long v = maskWords[(size_t)sh_i * memStride + w];
                s_globalSuppress[w] |= v;
            }
        }
        __syncthreads();  // Barrier B: suppression map updated

        // ---- Advance counters / position; early-exit if quota met ----
        if (threadIdx.x == 0) {
            if (sh_take) ++sh_keepCnt;
            if (sh_keepCnt >= maxOut) {
                sh_done = 1;
            } else {
                // Move to at least the following index so we don't revisit the same bit
                ++sh_i;
            }
        }
        __syncthreads();  // Barrier C: updated sh_keepCnt/sh_i/sh_done visible

        if (sh_done) break;
    }

    if (threadIdx.x == 0) {
        *outKeepCount = (sh_keepCnt < maxOut ? sh_keepCnt : maxOut);
    }
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

    int *           h_int;
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
    ws->h_int=(int *)cuda_malloc_host(4);

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
    cuda_free_host(ws->h_int);
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
        //const float* srcBatch = all_data + (size_t)b * rowSize * numBoxes;
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
            if (!data_is_fp16) {
                filterAndCollectKernel_BlockScan<<<b2,t2,0,stream>>>(
                    srcFloat, numBoxes, c, scoreThreshold,
                    ws->d_numCand,
                    ws->d_filteredIdx,
                    ws->d_filteredScores);
            } else {
                filterAndCollectKernelFP16_BlockScan<<<b2,t2,0,stream>>>(
                    srcHalf, numBoxes, c, scoreThreshold,
                    ws->d_numCand,
                    ws->d_filteredIdx,
                    ws->d_filteredScores);
            }
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(
                ws->h_int, ws->d_numCand, sizeof(int),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            h_numCand = ws->h_int[0];
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

            int dynStride = (h_numCand + 63) / 64;
            int memStride = ws->maxMaskStride;
            const int t3x = 128; // 4 warps per block
            const int warpsPerBlock = t3x / 32;
            dim3 threads3(t3x, 1);

            // blocks.x counts warps, not threads
            dim3 blocks3((h_numCand + warpsPerBlock - 1) / warpsPerBlock, dynStride);

            buildPairwiseMaskKernel<<<blocks3, threads3, 0, stream>>>(
                ws->d_corners,
                ws->d_filteredIdx,  // sortedIdx
                h_numCand,
                iouThreshold,
                dynStride,
                memStride,
                ws->d_mask + (size_t)c * ws->maxBoxes * memStride);
            CUDA_CHECK(cudaGetLastError());

            // IMPORTANT: shared memory sized by dynStride (NOT memStride)
            int sharedBytes = sizeof(unsigned long long) * dynStride;

            // Reasonable thread count: power-of-two up to 256
            auto nextPow2 = [](int x){ x--; x|=x>>1; x|=x>>2; x|=x>>4; x|=x>>8; x|=x>>16; return x+1; };
            int tSuppr = nextPow2(dynStride);
            tSuppr = tSuppr < 32 ? 32 : (tSuppr > 256 ? 256 : tSuppr);

            doSuppressionKernel<<<1, tSuppr, sharedBytes, stream>>>(
                ws->d_filteredIdx,
                ws->d_mask + (size_t)c * ws->maxBoxes * memStride,
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
