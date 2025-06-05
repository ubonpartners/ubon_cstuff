// cuda_nms_improved.cu

#include "cuda_nms.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>

//-------------------------------------------------------------------------------------------------
// Macro for CUDA error checking
//-------------------------------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)


//-------------------------------------------------------------------------------------------------
// Internal struct that contains all GPU buffers and limits.
//-------------------------------------------------------------------------------------------------
struct CudaNMSWorkspace_t {
    int           maxBoxes;
    int           maxClasses;
    int           maxOutPerClass;
    int           maxMaskStrideWords; // = ceil(maxBoxes / 64)

    cudaStream_t  stream;

    // GPU buffers:
    float*        d_corners;     // [maxBoxes × 4]
    float*        d_scoresTmp;   // [maxBoxes]
    int*          d_sortedIdx;   // [maxBoxes × maxClasses]
    unsigned long long* d_mask;  // [maxBoxes × maxMaskStrideWords × maxClasses]
    int*          d_keepCount;   // [maxClasses]
    int*          d_keptIdx;     // [maxClasses × maxOutPerClass]
};

//-------------------------------------------------------------------------------------------------
// KERNEL: Copy per-class scores (feature-major) into a contiguous [numBoxes] array.
//   Input layout: all_data[(4 + classIdx)*numBoxes + tid]
//   Output: out_scores[tid] = that score
//-------------------------------------------------------------------------------------------------
__global__ static void copyScoresKernel(
    const float* __restrict__ all_data,   // [rowSize × numBoxes]
    float*       __restrict__ out_scores,  // [numBoxes]
    int          numBoxes,
    int          classIdx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBoxes) return;
    out_scores[tid] = all_data[(4 + classIdx) * numBoxes + tid];
}

//-------------------------------------------------------------------------------------------------
// KERNEL: Zero‐out any scores < threshold (in place).
//   out_scores [numBoxes]
//-------------------------------------------------------------------------------------------------
__global__ static void applyThresholdKernel(
    float* __restrict__ out_scores,  // [numBoxes]
    int    numBoxes,
    float  scoreThreshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes) return;
    if (out_scores[idx] < scoreThreshold) {
        out_scores[idx] = -1e6f;
    }
}

//-------------------------------------------------------------------------------------------------
// KERNEL A: Convert (cx,cy,w,h) → (x1,y1,x2,y2) for up to N boxes.
//   in_centers [4 × numBoxes], out_corners [numBoxes × 4].
//   Layout: feature‐major, so in_centers[r * numBoxes + idx] gives r-th feature for box idx.
//   r = 0..3 correspond to cx, cy, w, h.
//-------------------------------------------------------------------------------------------------
__global__ static void centerToCornerKernel(
    const float* __restrict__ in_centers,  // [4 × numBoxes]
    float*       __restrict__ out_corners, // [numBoxes × 4]
    int          numBoxes
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBoxes) return;

    // Read feature-major layout:
    float cx = in_centers[0 * numBoxes + idx];
    float cy = in_centers[1 * numBoxes + idx];
    float w  = in_centers[2 * numBoxes + idx];
    float h  = in_centers[3 * numBoxes + idx];

    float x1 = cx - 0.5f * w;
    float y1 = cy - 0.5f * h;
    float x2 = cx + 0.5f * w;
    float y2 = cy + 0.5f * h;

    // Write box-major corners:
    out_corners[idx * 4 + 0] = x1;
    out_corners[idx * 4 + 1] = y1;
    out_corners[idx * 4 + 2] = x2;
    out_corners[idx * 4 + 3] = y2;
}

//-------------------------------------------------------------------------------------------------
// KERNEL B: Build pairwise IoU masks in 64‐box tiles.
//   corners     [numBoxes × 4]  (box‐major)
//   sortedIdx   [numBoxes]      (indices into corners/scores arrays)
//   numBoxes
//   iouThreshold
//   maskStrideWords = ceil(numBoxes / 64)
//   maskOut     [numBoxes × maskStrideWords] (uint64)
// Each (i,w) thread writes one 64-bit mask for box i vs. tile w.
//-------------------------------------------------------------------------------------------------
__global__ static void buildPairwiseMaskKernel(
    const float* __restrict__ corners,    // [numBoxes × 4]
    const int*   __restrict__ sortedIdx,  // [numBoxes]
    int          numBoxes,
    float        iouThreshold,
    int          maskStrideWords,
    unsigned long long* __restrict__ maskOut  // [numBoxes × maskStrideWords]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // box index
    int w = blockIdx.y;                             // which 64‐box tile

    if (i >= numBoxes || w >= maskStrideWords) return;

    int idx_i = sortedIdx[i];
    float x1_i = corners[idx_i * 4 + 0];
    float y1_i = corners[idx_i * 4 + 1];
    float x2_i = corners[idx_i * 4 + 2];
    float y2_i = corners[idx_i * 4 + 3];
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);

    unsigned long long mask = 0ULL;
    int startJ = w * 64;
    int endJ = startJ + 64;
    if (endJ > numBoxes) endJ = numBoxes;

    for (int j = startJ; j < endJ; ++j) {
        if (j <= i) continue;
        int idx_j = sortedIdx[j];
        float x1_j = corners[idx_j * 4 + 0];
        float y1_j = corners[idx_j * 4 + 1];
        float x2_j = corners[idx_j * 4 + 2];
        float y2_j = corners[idx_j * 4 + 3];
        float area_j = (x2_j - x1_j) * (y2_j - y1_j);

        float xx1 = fmaxf(x1_i, x1_j);
        float yy1 = fmaxf(y1_i, y1_j);
        float xx2 = fminf(x2_i, x2_j);
        float yy2 = fminf(y2_i, y2_j);
        float w_int = xx2 - xx1;
        float h_int = yy2 - yy1;
        if (w_int > 0.0f && h_int > 0.0f) {
            float inter = w_int * h_int;
            float ovr = inter / (area_i + area_j - inter);
            if (ovr > iouThreshold) {
                mask |= (1ULL << (j - startJ));
            }
        }
    }

    maskOut[i * maskStrideWords + w] = mask;
}

//-------------------------------------------------------------------------------------------------
// KERNEL C: Single-threaded suppression per class.
//   sortedIdx   [numBoxes]
//   maskWords   [numBoxes × maskStrideWords]
//   numBoxes
//   maskStrideWords
//   maxOut
//   outKeepCount  [1]
//   outKeepIdx    [maxOut]
// Uses dynamic shared memory of size maskStrideWords × sizeof(uint64).
// Only thread 0 in block does the work.
//-------------------------------------------------------------------------------------------------
__global__ static void doSuppressionKernel(
    const int* __restrict__ sortedIdx,                // [numBoxes]
    const unsigned long long* __restrict__ maskWords, // [numBoxes × maskStrideWords]
    int                    numBoxes,
    int                    maskStrideWords,
    int                    maxOut,
    int* __restrict__      outKeepCount,         // scalar
    int* __restrict__      outKeepIdx            // [maxOut]
) {
    extern __shared__ unsigned long long s_globalSuppress[];

    // Only one thread does the work
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Initialize suppression bitmask
    for (int w = 0; w < maskStrideWords; ++w) {
        s_globalSuppress[w] = 0ULL;
    }

    int keepCnt = 0;
    for (int i = 0; i < numBoxes; ++i) {
        int wordIdx = i >> 6;
        int bitIdx  = i & 63;
        unsigned long long m = s_globalSuppress[wordIdx];
        if ((m & (1ULL << bitIdx)) == 0ULL) {
            if (keepCnt < maxOut) {
                outKeepIdx[keepCnt] = sortedIdx[i];
            }
            keepCnt++;
            for (int w = 0; w < maskStrideWords; ++w) {
                s_globalSuppress[w] |= maskWords[i * maskStrideWords + w];
            }
        }
    }

    *outKeepCount = (keepCnt < maxOut ? keepCnt : maxOut);
}

//-------------------------------------------------------------------------------------------------
// Allocate a new workspace and return its handle.
//-------------------------------------------------------------------------------------------------
CudaNMSHandle cuda_nms_allocate_workspace(
    int maxBoxes,
    int maxClasses,
    int maxOutPerClass,
    cudaStream_t stream
) {
    assert(maxBoxes > 0 && maxClasses > 0 && maxOutPerClass > 0);

    CudaNMSWorkspace_t* ws = (CudaNMSWorkspace_t*)std::malloc(sizeof(CudaNMSWorkspace_t));
    if (!ws) return nullptr;

    ws->maxBoxes       = maxBoxes;
    ws->maxClasses     = maxClasses;
    ws->maxOutPerClass = maxOutPerClass;
    ws->stream         = stream;

    // Compute mask stride (in 64-bit words) for the largest possible numBoxes:
    ws->maxMaskStrideWords = (maxBoxes + 63) / 64;

    // Allocate GPU buffers:
    CUDA_CHECK(cudaMalloc(&ws->d_corners,   sizeof(float) * (size_t)maxBoxes * 4));
    CUDA_CHECK(cudaMalloc(&ws->d_scoresTmp, sizeof(float) * (size_t)maxBoxes));
    CUDA_CHECK(cudaMalloc(&ws->d_sortedIdx, sizeof(int)   * (size_t)maxBoxes * maxClasses));
    CUDA_CHECK(cudaMalloc(&ws->d_mask,      sizeof(unsigned long long) *
                                        (size_t)maxBoxes * ws->maxMaskStrideWords * maxClasses));
    CUDA_CHECK(cudaMalloc(&ws->d_keepCount, sizeof(int)   * (size_t)maxClasses));
    CUDA_CHECK(cudaMalloc(&ws->d_keptIdx,   sizeof(int)   * (size_t)maxClasses * maxOutPerClass));

    return (CudaNMSHandle)ws;
}

//-------------------------------------------------------------------------------------------------
// Free all GPU buffers and the struct itself.
//-------------------------------------------------------------------------------------------------
void cuda_nms_free_workspace(CudaNMSHandle handle) {
    if (!handle) return;
    CudaNMSWorkspace_t* ws = (CudaNMSWorkspace_t*)handle;

    CUDA_CHECK(cudaFree(ws->d_corners));
    CUDA_CHECK(cudaFree(ws->d_scoresTmp));
    CUDA_CHECK(cudaFree(ws->d_sortedIdx));
    CUDA_CHECK(cudaFree(ws->d_mask));
    CUDA_CHECK(cudaFree(ws->d_keepCount));
    CUDA_CHECK(cudaFree(ws->d_keptIdx));

    std::free(ws);
}

//-------------------------------------------------------------------------------------------------
// cuda_nms_run
//   Uses only buffers inside the handle; no further allocations.
//   Expects input as [batch, rowSize, numBoxes] in feature-major layout.
//-------------------------------------------------------------------------------------------------

void cuda_nms_run(
    CudaNMSHandle                 handle,
    const float*                  all_data,      // [batch × rowSize × numBoxes], feature‐major
    int                           batch_size,
    int                           numBoxes,
    int                           numClasses,
    int                           rowSize,
    float                         scoreThreshold,
    float                         iouThreshold,
    int                           maxOutPerClass,
    std::vector<std::vector<int>>& keptIndices,
    cudaStream_t                  stream
) {
    assert(handle);
    CudaNMSWorkspace_t* ws = (CudaNMSWorkspace_t*)handle;

    // Validate dimensions:
    assert(batch_size > 0);
    assert(numBoxes > 0 && numBoxes <= ws->maxBoxes);
    assert(numClasses > 0 && numClasses <= ws->maxClasses);
    assert(rowSize >= 4 + numClasses);
    assert(maxOutPerClass > 0 && maxOutPerClass <= ws->maxOutPerClass);

    if (stream == nullptr) stream = ws->stream;

    // Resize output: flattened vector of size [batch_size × numClasses]
    keptIndices.resize((size_t)batch_size * numClasses);
    for (auto& v : keptIndices) v.clear();

    // Pointers to GPU buffers:
    float*        d_corners   = ws->d_corners;     // [maxBoxes × 4]
    float*        d_scoresTmp = ws->d_scoresTmp;   // [maxBoxes]
    int*          d_sortedIdx = ws->d_sortedIdx;   // [maxBoxes × maxClasses]
    unsigned long long* d_mask = ws->d_mask;       // [maxBoxes × maxMaskStrideWords × maxClasses]
    int*          d_keepCount = ws->d_keepCount;   // [maxClasses]
    int*          d_keptIdx   = ws->d_keptIdx;     // [maxClasses × maxOutPerClass]

    // Host‐side temp buffers for filtering/sorting:
    std::vector<float>   h_scores(numBoxes);
    std::vector<std::pair<float,int>> filtered;        // ★ will hold only (score,index) ≥ threshold
    std::vector<int>     h_sortedIdx(numBoxes);

    // For each batch:
    for (int b = 0; b < batch_size; ++b) {
        // “feature‐major” pointer into batch b
        const float* srcBatch = all_data + (size_t)b * (rowSize * numBoxes);

        // 1) Convert centers → corners:
        {
            int threads = 256;
            int blocks = (numBoxes + threads - 1) / threads;
            centerToCornerKernel<<<blocks, threads, 0, stream>>>(
                srcBatch,     // points to “cx” for each of the numBoxes
                d_corners,
                numBoxes
            );
            CUDA_CHECK(cudaGetLastError());
        }

        // 2) For each class c, copy, threshold, filter, sort, build mask, suppress:
        for (int c = 0; c < numClasses; ++c) {
            // 2.a) Copy per‐class scores into d_scoresTmp:
            {
                int threads = 256;
                int blocks = (numBoxes + threads - 1) / threads;
                copyScoresKernel<<<blocks, threads, 0, stream>>>(
                    srcBatch,
                    d_scoresTmp,
                    numBoxes,
                    c
                );
                CUDA_CHECK(cudaGetLastError());
            }

            // 2.b) Apply threshold so that any score < scoreThreshold becomes −1e6f
            {
                int threads = 256;
                int blocks = (numBoxes + threads - 1) / threads;
                applyThresholdKernel<<<blocks, threads, 0, stream>>>(
                    d_scoresTmp,
                    numBoxes,
                    scoreThreshold
                );
                CUDA_CHECK(cudaGetLastError());
            }

            // 2.c) Copy scoresTmp back to host for filtering:
            CUDA_CHECK(cudaMemcpyAsync(
                h_scores.data(),
                d_scoresTmp,
                sizeof(float) * (size_t)numBoxes,
                cudaMemcpyDeviceToHost,
                stream
            ));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // ★ 2.d) Build a list of only those (score,idx) where score ≥ threshold
            filtered.clear();
            filtered.reserve(numBoxes);
            for (int i = 0; i < numBoxes; ++i) {
                if (h_scores[i] >= scoreThreshold) {
                    filtered.emplace_back(h_scores[i], i);
                }
            }

            // If no box passes the threshold, skip this class entirely
            if (filtered.empty()) {
                // Zero out keep count on device for class c:
                int zero = 0;
                CUDA_CHECK(cudaMemcpyAsync(
                    d_keepCount + c,
                    &zero,
                    sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream
                ));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                // keptIndices[b * numClasses + c] stays empty
                continue;
            }

            // ★ 2.e) Sort ONLY the filtered list by descending score
            std::sort(
                filtered.begin(),
                filtered.end(),
                [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                    return a.first > b.first;
                }
            );

            // ★ 2.f) Build host‐sorted index array of “valid” boxes only
            int numCandidates = (int)filtered.size();
            for (int i = 0; i < numCandidates; ++i) {
                h_sortedIdx[i] = filtered[i].second;
            }

            // 2.g) Copy that truncated sortedIdx (numCandidates entries) up to device
            {
                int* d_sortedForClass = d_sortedIdx + (size_t)c * ws->maxBoxes;
                CUDA_CHECK(cudaMemcpyAsync(
                    d_sortedForClass,
                    h_sortedIdx.data(),
                    sizeof(int) * (size_t)numCandidates,
                    cudaMemcpyHostToDevice,
                    stream
                ));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // 3) Build pairwise IoU mask over exactly numCandidates “valid” boxes:
            {
                int* d_sortedForClass = d_sortedIdx + (size_t)c * ws->maxBoxes;
                unsigned long long* d_maskForClass = d_mask +
                    (size_t)c * ws->maxBoxes * ws->maxMaskStrideWords;

                int validMaskStride = (numCandidates + 63) / 64;
                dim3 threads2(32, 1);
                dim3 blocks2((numCandidates + 31) / 32, validMaskStride);
                buildPairwiseMaskKernel<<<blocks2, threads2, 0, stream>>>(
                    d_corners,
                    d_sortedForClass,
                    numCandidates,
                    iouThreshold,
                    validMaskStride,
                    d_maskForClass
                );
                CUDA_CHECK(cudaGetLastError());
            }

            // 4) Suppression pass over exactly numCandidates “valid” boxes:
            {
                int* d_sortedForClass = d_sortedIdx + (size_t)c * ws->maxBoxes;
                unsigned long long* d_maskForClass = d_mask +
                    (size_t)c * ws->maxBoxes * ws->maxMaskStrideWords;
                int* d_keepCntForClass = d_keepCount + c;
                int* d_keptIdxForClass = d_keptIdx + (size_t)c * ws->maxOutPerClass;

                int validMaskStride = (numCandidates + 63) / 64;
                int sharedSuppressBytes = sizeof(unsigned long long) * validMaskStride;

                doSuppressionKernel<<<1, 1, sharedSuppressBytes, stream>>>(
                    d_sortedForClass,
                    d_maskForClass,
                    numCandidates,     // only suppress among these many boxes
                    validMaskStride,
                    maxOutPerClass,
                    d_keepCntForClass,
                    d_keptIdxForClass
                );
                CUDA_CHECK(cudaGetLastError());

                // Copy back keep count
                int hostKeepCnt = 0;
                CUDA_CHECK(cudaMemcpyAsync(
                    &hostKeepCnt,
                    d_keepCntForClass,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (hostKeepCnt > maxOutPerClass) hostKeepCnt = maxOutPerClass;

                // Copy back exactly hostKeepCnt indices
                if (hostKeepCnt > 0) {
                    std::vector<int> tmp(hostKeepCnt);
                    CUDA_CHECK(cudaMemcpyAsync(
                        tmp.data(),
                        d_keptIdxForClass,
                        sizeof(int) * (size_t)hostKeepCnt,
                        cudaMemcpyDeviceToHost,
                        stream
                    ));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    keptIndices[(size_t)b * numClasses + c] = std::move(tmp);
                }
            }

        } // end per‐class loop
    } // end per‐batch loop
}


#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cassert>
#include <cstdio>

//-------------------------------------------------------------------------------------------------
// Helper for CUDA error checking
//-------------------------------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)


//-------------------------------------------------------------------------------------------------
// Kernel: gatherFeatures
//
// For each “kept” entry k ∈ [0..totalKept−1] and each feature r ∈ [0..rowSize−1], copy
//    dst[k * rowSize + r] = src[r * numBoxes + boxIdx[k]]
//
// Inputs:
//   src           [ rowSize × numBoxes ]   (feature-major layout on device)
//   keptIdx       [ totalKept ]            (array of box indices to gather, on device)
//   numBoxes      total number of boxes
//   rowSize       number of features per box
//   totalKept     total number of kept indices (length of keptIdx[])
// Output:
//   dst           [ totalKept × rowSize ]  (gathered output, on device)
//
// We launch with a 2D grid:
//   blockDim = (32, 32)  or similar
//   gridDim  = ( (totalKept+31)/32, (rowSize+31)/32 )
//
__global__ static void gatherFeaturesKernel(
    const float* __restrict__ src,        // [ rowSize × numBoxes ]
    const int*   __restrict__ keptIdx,    // [ totalKept ]
    int          numBoxes,
    int          rowSize,
    int          totalKept,
    float*       __restrict__ dst         // [ totalKept × rowSize ]
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // which kept index
    int r = blockIdx.y * blockDim.y + threadIdx.y; // which feature

    if (k >= totalKept || r >= rowSize) return;

    int box = keptIdx[k]; // original box index ∈ [0..numBoxes−1]
    // Copy the feature r for that box into the output slot (k, r)
    dst[k * rowSize + r] = src[r * numBoxes + box];
}


//-------------------------------------------------------------------------------------------------
// gatherKeptOutputs
//
// Given:
//   - deviceOutputDev:  pointer to the original full-output buffer on device, shaped [ rowSize × numBoxes ]
//   - numBoxes, rowSize: as above
//   - keptIndices:     std::vector<std::vector<int>>, with one inner vector per class (or however you grouped them).
//                      We will concatenate all of them in order: keptIndices[0], then keptIndices[1], etc.
// Returns:
//   - hostGathered:    a std::vector<float> of length (totalKept × rowSize), holding the features for each kept index,
//                      in the same class-ordered sequence as in keptIndices.
//
// Note: We assume batch_size = 1. If you want multiple batches, you can call this per-batch.
//-------------------------------------------------------------------------------------------------
void cuda_nms_gather_kept_outputs(
    const float*                             deviceOutputDev,   // [ rowSize × numBoxes ], on device
    int                                      numBoxes,
    int                                      rowSize,
    const std::vector<std::vector<int>>&     keptIndices,
    std::vector<float>&                      hostGathered     // (output) length = totalKept × rowSize
) {
    // 1) Flatten keptIndices into a single host array keptFlat[]
    int totalClasses = (int) keptIndices.size();
    int totalKept = 0;
    for (int c = 0; c < totalClasses; ++c) {
        totalKept += (int) keptIndices[c].size();
    }

    // Early-out if nothing was kept:
    if (totalKept == 0) {
        hostGathered.clear();
        return;
    }

    std::vector<int> keptFlat;
    keptFlat.reserve(totalKept);
    for (int c = 0; c < totalClasses; ++c) {
        for (int idx : keptIndices[c]) {
            // Sanity check:
            assert(idx >= 0 && idx < numBoxes);
            keptFlat.push_back(idx);
        }
    }
    assert((int)keptFlat.size() == totalKept);

    // 2) Allocate device-side buffer for keptFlat indices
    int* d_keptFlat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keptFlat, sizeof(int) * (size_t)totalKept));
    CUDA_CHECK(cudaMemcpy(
        d_keptFlat,
        keptFlat.data(),
        sizeof(int) * (size_t)totalKept,
        cudaMemcpyHostToDevice
    ));

    // 3) Allocate device-side output buffer [ totalKept × rowSize ] floats
    float* d_gathered = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gathered, sizeof(float) * (size_t)totalKept * (size_t)rowSize));

    // 4) Launch gatherFeaturesKernel
    //    We'll use a 2D block of (32 × 32) threads for example (tweak as needed).
    const int TILE_X = 32;
    const int TILE_Y = 32;

    dim3 blockDim(TILE_X, TILE_Y);
    dim3 gridDim(
        (totalKept + TILE_X - 1) / TILE_X,
        (rowSize   + TILE_Y - 1) / TILE_Y
    );

    gatherFeaturesKernel<<<gridDim, blockDim>>>(
        deviceOutputDev,
        d_keptFlat,
        numBoxes,
        rowSize,
        totalKept,
        d_gathered
    );
    CUDA_CHECK(cudaGetLastError());

    // 5) Copy the gathered results back to host
    hostGathered.resize((size_t)totalKept * (size_t)rowSize);
    CUDA_CHECK(cudaMemcpy(
        hostGathered.data(),
        d_gathered,
        sizeof(float) * (size_t)totalKept * (size_t)rowSize,
        cudaMemcpyDeviceToHost
    ));

    // 6) Free device scratch
    CUDA_CHECK(cudaFree(d_keptFlat));
    CUDA_CHECK(cudaFree(d_gathered));
}
