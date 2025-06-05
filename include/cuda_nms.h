#pragma once

#include <cuda_runtime.h>
#include <vector>

/**
 * cuda_nms: a minimal, standalone “Efficient NMS” for dynamic box‐counts and dynamic class‐counts,
 *            using an opaque workspace handle.
 *
 * Typical usage:
 *
 *   // 1. Allocate once, specifying the maximum sizes you’ll ever need:
 *   CudaNMSHandle handle = cuda_nms_allocate_workspace(
 *       5000,   // maxBoxes
 *       10,     // maxClasses
 *       500,    // maxOutPerClass
 *       nullptr // stream
 *   );
 *
 *   // 2. For each inference, call:
 *   std::vector<std::vector<int>> keptIndices;
 *   cuda_nms_run(
 *       handle,
 *       hostAllDataPtr,  // [batch_size × numBoxes × rowSize] floats on host
 *       batch_size,      // e.g. 1
 *       numBoxes,        // e.g. 1000  (≤ maxBoxes)
 *       numClasses,      // e.g. 3     (≤ maxClasses)
 *       rowSize,         // e.g. 4 + numClasses + extra  (≥ 4 + numClasses)
 *       0.05f,           // scoreThreshold
 *       0.45f,           // iouThreshold
 *       maxOutPerClass,  // ≤ maxOutPerClass
 *       keptIndices,
 *       nullptr          // stream
 *   );
 *
 *   // 3. When done, free the workspace:
 *   cuda_nms_free_workspace(handle);
 *
 * Functions:
 *
 * ▶ cuda_nms_allocate_workspace(maxBoxes, maxClasses, maxOutPerClass, stream)
 *     • Returns a handle for GPU buffers sized to cover up to maxBoxes, maxClasses, maxOutPerClass.
 *     • You must call this before any cuda_nms_run. If your run‐time parameters exceed these maxima, it will assert/fail.
 *
 * ▶ cuda_nms_run(handle, all_data, batch_size, numBoxes, numClasses, rowSize,
 *                scoreThreshold, iouThreshold, maxOutPerClass, keptIndices, stream)
 *     • Uses only the GPU buffers inside handle—no further allocation.
 *     • all_data points to [batch_size × numBoxes × rowSize] floats on the host.
 *       For each box i in batch b:
 *         all_data[((b*numBoxes + i) * rowSize) + 0..3]               = {cx, cy, w, h}
 *         all_data[((b*numBoxes + i) * rowSize) + 4..(4+numClasses−1)] = per‐class scores
 *         the rest of each row (rowSize − (4+numClasses) floats) is ignored.
 *     • Outputs keptIndices resized to [batch_size × numClasses], each a std::vector<int> of kept box indices.
 *
 * ▶ cuda_nms_free_workspace(handle)
 *     • Frees all GPU buffers associated with handle. After this call, handle is invalid.
 *
 * All functions accept an optional cudaStream_t. If you pass NULL, they use the default stream.
 */

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to the workspace struct
typedef struct CudaNMSWorkspace_t* CudaNMSHandle;

/**
 * Allocate GPU workspace buffers.
 *   maxBoxes:        maximum number of boxes per batch
 *   maxClasses:      maximum number of classes
 *   maxOutPerClass:  maximum boxes to keep per class
 *   stream:          optional CUDA stream (default = NULL)
 *
 * Returns a non-null handle on success, which must later be freed with cuda_nms_free_workspace.
 */
CudaNMSHandle cuda_nms_allocate_workspace(
    int           maxBoxes,
    int           maxClasses,
    int           maxOutPerClass,
    cudaStream_t  stream  // optional; default = NULL
);

/**
 * Run NMS using only the GPU buffers in handle.
 *   handle:            previously returned by cuda_nms_allocate_workspace
 *   all_data:          host pointer to [batch_size × numBoxes × rowSize] floats
 *   batch_size:        number of batches (≥1)
 *   numBoxes:          boxes per batch (≤ maxBoxes)
 *   numClasses:        classes (≤ maxClasses)
 *   rowSize:           floats per row (≥ 4 + numClasses)
 *   scoreThreshold:    drop boxes below this before NMS
 *   iouThreshold:      IoU threshold for suppression
 *   maxOutPerClass:    kept‐boxes limit (≤ maxOutPerClass)
 *   keptIndices:       output container, resized to [batch_size × numClasses]
 *   stream:            optional CUDA stream (default = NULL)
 *
 * After return, keptIndices[b * numClasses + c] holds the kept box indices (0..numBoxes−1).
 */
void cuda_nms_run(
    CudaNMSHandle                 handle,
    const float*                  all_data,
    int                           batch_size,
    int                           numBoxes,
    int                           numClasses,
    int                           rowSize,
    float                         scoreThreshold,
    float                         iouThreshold,
    int                           maxOutPerClass,
    std::vector<std::vector<int>>& keptIndices,
    cudaStream_t                  stream  // optional; default = NULL
);

void cuda_nms_gather_kept_outputs(
    const float*                             deviceOutputDev,   // [ rowSize × numBoxes ], on device
    int                                      numBoxes,
    int                                      rowSize,
    const std::vector<std::vector<int>>&     keptIndices,
    std::vector<float>&                      hostGathered     // (output) length = totalKept × rowSize
);

/**
 * Free all GPU buffers associated with handle.
 * After this, handle is no longer valid.
 */
void cuda_nms_free_workspace(CudaNMSHandle handle);

#ifdef __cplusplus
}  // extern "C"
#endif
