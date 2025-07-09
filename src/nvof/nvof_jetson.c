#include "log.h"
#include "cuda_stuff.h"
#include "nvof.h"
#include "display.h"
#include <stdio.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <mutex>

#if (UBONCSTUFF_PLATFORM == 1) // Orin Nano
struct nvof
{
    int max_width;
    int max_height;
    int width;
    int height;
    int gridsize;
    int outW;
    int outH;

    nvof_results_t results;
};


nvof_results_t *nvof_execute(nvof_t *v, image_t *img_in)
{
    (void)v; (void)img_in;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return NULL;
}

nvof_t *nvof_create(void *context, int max_width, int max_height)
{
    (void)context; (void)max_width; (void)max_height;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return NULL;
}

void nvof_destroy(nvof_t *n)
{
    (void)n;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return;
}
#endif //(UBONCSTUFF_PLATFORM == 1)

#if 0 // chat GPU version

// nvof_vpi.cpp – Jetson Orin Nano implementation of the NVOF interface using VPI Dense Optical Flow
// -----------------------------------------------------------------------------------------------
// Build example (Jetson ‑ JetPack 6+):
//   g++ -std=c++17 -I/usr/include/vpi -I/path/to/your/includes -DU BUNCSTUFF_PLATFORM=1 \
//       nvof_vpi.cpp -lvpi -lcuda -o libnvof_vpi.so -fPIC -shared
// ------------------------------------------------------------------------------------------------

#include "nvof.h"
#include "log.h"          // keep your existing logging helpers
#include "cuda_stuff.h"   // create_cuda_stream()/destroy_cuda_stream() helpers
#include "display.h"      // if you already rely on it elsewhere

#include <vpi/vpi.h>
#include <cassert>
#include <cstring>
#include <mutex>

#if (UBONCSTUFF_PLATFORM == 1) // Jetson / VPI back‑end

// -----------------------------------------------------------------------------------------------
// Convenience error‑handling helpers
// -----------------------------------------------------------------------------------------------
#define CHECK_VPI(call)                                                                            \
    do                                                                                            \
    {                                                                                             \
        VPIStatus _status = (call);                                                               \
        if (_status != VPI_SUCCESS)                                                               \
        {                                                                                         \
            const char *errStr;                                                                   \
            vpiGetLastStatusMessage(&errStr);                                                     \
            log_fatal("VPI error %d: %s", _status, errStr ? errStr : "<unknown>");             \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

// -----------------------------------------------------------------------------------------------
// Internal structure backing the opaque handle
// -----------------------------------------------------------------------------------------------
struct nvof
{
    // configuration / limits
    int max_width  = 0;
    int max_height = 0;

    // current working size (<= max_*)
    int width  = 0;
    int height = 0;
    int gridsize = 4;             // fixed to 4×4 like original code
    int outW = 0;                 // width  of flow grid = ceil(width/4)
    int outH = 0;                 // height of flow grid

    // frame counter (used to skip first execution)
    uint32_t count = 0;

    // VPI resources
    VPIStream           stream  = nullptr; // CUDA backend stream (we reuse the CUDA stream we already have)
    VPIOpticalFlow      of      = nullptr; // algorithm instance
    VPIImage            imgCur  = nullptr; // current frame
    VPIImage            imgPrev = nullptr; // previous frame
    VPIImage            flowImg = nullptr; // 2‑channel S10.5 flow field

    CUstream            cuStream = nullptr; // actual CUDA stream we pass to VPI (optional but keeps symmetry with desktop path)

    // host‑side buffers returned to the user
    flow_vector_t *flowBufHost = nullptr;

    nvof_results_t results;      // pre‑filled each execute()
};

// -----------------------------------------------------------------------------------------------
// Utility: destroy VPI image if allocated
// -----------------------------------------------------------------------------------------------
static void destroy_vpi_image(VPIImage &img)
{
    if (img)
    {
        vpiImageDestroy(img);
        img = nullptr;
    }
}

// -----------------------------------------------------------------------------------------------
// Utility: (re‑)initialise VPI resources for a given resolution
// -----------------------------------------------------------------------------------------------
static void vpi_set_size(nvof *v, int width, int height)
{
    if (v->width == width && v->height == height)
        return; // nothing to do

    // destroy any existing size‑dependent resources
    destroy_vpi_image(v->imgCur);
    destroy_vpi_image(v->imgPrev);
    destroy_vpi_image(v->flowImg);

    if (v->of)
    {
        vpiOpticalFlowDestroy(v->of);
        v->of = nullptr;
    }

    // set new dims
    v->width  = width;
    v->height = height;
    v->outW   = (width  + v->gridsize - 1) / v->gridsize;
    v->outH   = (height + v->gridsize - 1) / v->gridsize;

    // -------- Create algorithm instance --------------------------------------------------------
    VPIOpticalFlowDenseParams ofParams;
    vpiInitOpticalFlowDenseParams(&ofParams);
    ofParams.gridSize      = v->gridsize;           // 4×4 output grid
    ofParams.hintPrecision = VPI_HINT_PRECISION_NONE;
    ofParams.enableHints   = 0;

    // Use CUDA backend (hardware path on Orin)
    CHECK_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_CUDA, width, height, VPI_IMAGE_FORMAT_NV12_ER,
                                        &ofParams, &v->of));

    // -------- Allocate VPI images --------------------------------------------------------------
    CHECK_VPI(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_NV12_ER, 0, &v->imgCur));
    CHECK_VPI(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_NV12_ER, 0, &v->imgPrev));

    // Flow image is 2× int16 in S10.5 (matches nvof_t expectation)
    CHECK_VPI(vpiImageCreate(v->outW, v->outH, VPI_IMAGE_FORMAT_2S16, 0, &v->flowImg));

    // -------- Host buffer for results ----------------------------------------------------------
    if (v->flowBufHost)
        cuMemFreeHost(v->flowBufHost);

    cuMemAllocHost(reinterpret_cast<void **>(&v->flowBufHost), sizeof(flow_vector_t) * v->outW * v->outH);
    std::memset(v->flowBufHost, 0, sizeof(flow_vector_t) * v->outW * v->outH);
}

// -----------------------------------------------------------------------------------------------
// Public API – nvof_execute
// -----------------------------------------------------------------------------------------------
nvof_results_t *nvof_execute(nvof *v, image_t *img_in)
{
    assert(img_in != nullptr);

    // --- Determine the operating resolution ----------------------------------------------------
    int w = v->max_width & (~3);
    int h = ((static_cast<long long>(w) * img_in->height) / img_in->width) & (~3);
    if (h > v->max_height)
    {
        h = v->max_height & (~3);
        w = ((static_cast<long long>(h) * img_in->width) / img_in->height) & (~3);
    }

    vpi_set_size(v, w, h);

    // --- Pre‑process: scale and convert to NV12 resident on the GPU -----------------------------
    image_t *scaled = image_scale(img_in, v->width, v->height);
    image_t *img    = image_convert(scaled, IMAGE_FORMAT_NV12_DEVICE);

    // --- Copy CUDA memory into VPI image --------------------------------------------------------
    // Wrap destination image plane for direct device copy (VPI offers handy helper for NV12)
    VPIImageData imgDataCur;
    CHECK_VPI(vpiImageLock(v->imgCur, VPI_LOCK_WRITE | VPI_LOCK_TYPE_CUDA, &imgDataCur));

    const CUdeviceptr dstY  = reinterpret_cast<CUdeviceptr>(imgDataCur.buffer.cuda.buffer);
    const size_t      dstPY = imgDataCur.buffer.cuda.pitchBytes[0];
    const CUdeviceptr dstUV = reinterpret_cast<CUdeviceptr>(imgDataCur.buffer.cuda.buffer) + imgDataCur.buffer.cuda.offset[1];
    const size_t      dstPU = imgDataCur.buffer.cuda.pitchBytes[1];

    // Plane Y copy
    CUDA_MEMCPY2D cpy{};
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.srcDevice     = reinterpret_cast<CUdeviceptr>(img->device_mem);
    cpy.srcPitch      = img->stride_y;
    cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.dstDevice     = dstY;
    cpy.dstPitch      = dstPY;
    cpy.WidthInBytes  = img->width;
    cpy.Height        = img->height;
    CHECK_CUDA_CALL(cuMemcpy2DAsync(&cpy, v->cuStream));

    // Plane UV copy (immediately after Y in NV12 layout)
    cpy.srcDevice    = reinterpret_cast<CUdeviceptr>(img->device_mem) + img->stride_y * img->height;
    cpy.srcPitch     = img->stride_uv;
    cpy.dstDevice    = dstUV;
    cpy.dstPitch     = dstPU;
    cpy.Height       = img->height / 2; // NV12 chroma plane is half height
    CHECK_CUDA_CALL(cuMemcpy2DAsync(&cpy, v->cuStream));

    // Done – unlock (defered until copies finish). The stream dependency is implicit as we used same CUstream.
    CHECK_VPI(vpiImageUnlock(v->imgCur));

    // --- Run optical flow (skip first frame) ----------------------------------------------------
    if (v->count > 0)
    {
        CHECK_VPI(vpiSubmitOpticalFlowDense(v->stream, VPI_BACKEND_CUDA, v->of, v->imgPrev, v->imgCur, v->flowImg, nullptr));
    }

    // --- Synchronise (wait for all work on stream) ----------------------------------------------
    CHECK_VPI(vpiStreamSync(v->stream));

    // --- Retrieve results ----------------------------------------------------------------------
    if (v->count > 0)
    {
        // lock for reading – gives us device pointer + pitch; we copy into host‑pinned buffer
        VPIImageData flowData;
        CHECK_VPI(vpiImageLock(v->flowImg, VPI_LOCK_READ | VPI_LOCK_TYPE_CUDA, &flowData));

        const size_t pitch = flowData.buffer.cuda.pitchBytes[0];
        const CUdeviceptr srcFlow = reinterpret_cast<CUdeviceptr>(flowData.buffer.cuda.buffer);

        CUDA_MEMCPY2D cpyFlow{};
        cpyFlow.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cpyFlow.srcDevice     = srcFlow;
        cpyFlow.srcPitch      = pitch;
        cpyFlow.dstMemoryType = CU_MEMORYTYPE_HOST;
        cpyFlow.dstHost       = v->flowBufHost;
        cpyFlow.dstPitch      = sizeof(flow_vector_t) * v->outW;
        cpyFlow.WidthInBytes  = sizeof(flow_vector_t) * v->outW;
        cpyFlow.Height        = v->outH;
        CHECK_CUDA_CALL(cuMemcpy2DAsync(&cpyFlow, v->cuStream));
        CHECK_CUDA_CALL(cuStreamSynchronize(v->cuStream));

        CHECK_VPI(vpiImageUnlock(v->flowImg));
    }
    else
    {
        // first call – no flow available yet, return zeros
        std::memset(v->flowBufHost, 0, sizeof(flow_vector_t) * v->outW * v->outH);
    }

    // --- Prepare return struct -----------------------------------------------------------------
    v->results.grid_w = v->outW;
    v->results.grid_h = v->outH;
    v->results.costs  = nullptr;          // user said they don't need cost surface
    v->results.flow   = v->flowBufHost;

    // --- Book‑keeping: swap current/previous ----------------------------------------------------
    std::swap(v->imgCur, v->imgPrev);
    v->count++;

    destroy_image(scaled);
    destroy_image(img);

    return &v->results;
}

// -----------------------------------------------------------------------------------------------
// Public API – create/destroy
// -----------------------------------------------------------------------------------------------
nvof *nvof_create(void * /*context*/, int max_width, int max_height)
{
    check_cuda_inited(); // reuse your existing CUDA initialisation helper

    auto *n = static_cast<nvof *>(std::calloc(1, sizeof(nvof)));
    if (!n)
        return nullptr;

    n->max_width  = max_width;
    n->max_height = max_height;
    n->gridsize   = 4; // fixed like desktop path

    // create CUDA stream reused both by our code and VPI
    n->cuStream = create_cuda_stream();
    CHECK_VPI(vpiStreamCreateWrapperCUDA(n->cuStream, 0, &n->stream));

    // defer actual resource creation until the first execute() call (when we know dimensions)

    return n;
}

void nvof_destroy(nvof *n)
{
    if (!n)
        return;

    if (n->stream)
        vpiStreamDestroy(n->stream);

    destroy_vpi_image(n->imgCur);
    destroy_vpi_image(n->imgPrev);
    destroy_vpi_image(n->flowImg);

    if (n->of)
        vpiOpticalFlowDestroy(n->of);

    if (n->flowBufHost)
        cuMemFreeHost(n->flowBufHost);

    destroy_cuda_stream(n->cuStream);

    std::free(n);
}

#endif // UBONCSTUFF_PLATFORM == 1
#endif
