// nvof_vpi.cpp – Jetson Orin Nano implementation of the NVOF interface using VPI Dense Optical Flow

#if (UBONCSTUFF_PLATFORM == 1)
// nvof_vpi.cpp  –  Jetson Orin Nano implementation using VPI-3 Dense Optical Flow
#include "nvof.h"
#include "log.h"
#include "cuda_stuff.h"
#include "display.h"

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/CUDAInterop.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <cstring>
#include <mutex>

#define CHECK_VPI(call)                                                     \
    do {                                                                    \
        VPIStatus _status = (call);                                         \
        if (_status != VPI_SUCCESS) {                                       \
            log_fatal("VPI call \"%s\" failed (status=%d)", #call, _status);\
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// -------------------------------------------------------------------------
// struct nvof (unchanged except for include order)
// -------------------------------------------------------------------------
struct nvof
{
    int  max_width  = 0, max_height = 0;
    int  width      = 0, height     = 0;
    int  gridsize   = 4, outW = 0,   outH = 0;
    uint32_t count  = 0;

    flow_vector_t *flowBufHost = nullptr;
    uint8_t       *costBufHost = nullptr;

    CUstream  cuStream  = nullptr;
    VPIStream vpiStream = nullptr;

    VPIPayload ofPayload = nullptr;
    VPIImage   curImg    = nullptr;
    VPIImage   prevImg   = nullptr;
    VPIImage   mvImg     = nullptr;

    bool use_nv12 = true;

    nvof_results_t results{};
};

// -------------------------------------------------------------------------
// helpers
// -------------------------------------------------------------------------
static inline void destroy_vpi_image(VPIImage &img)
{
    if (img) { vpiImageDestroy(img); img = nullptr; }
}

static inline void destroy_host_buffers(nvof *v)
{
    if (v->flowBufHost) { cudaFreeHost(v->flowBufHost); v->flowBufHost = nullptr; }
    if (v->costBufHost) { cudaFreeHost(v->costBufHost); v->costBufHost = nullptr; }
}

static void nvof_set_size(nvof *v, int w, int h)
{
    if (v->width == w && v->height == h) return;

    destroy_vpi_image(v->curImg);
    destroy_vpi_image(v->prevImg);
    destroy_vpi_image(v->mvImg);
    if (v->ofPayload) { vpiPayloadDestroy(v->ofPayload); v->ofPayload = nullptr; }
    destroy_host_buffers(v);

    v->width  = w;
    v->height = h;
    v->gridsize = 4;
    v->outW = (w + 3) / 4;
    v->outH = (h + 3) / 4;

    const VPIImageFormat inFmt = v->use_nv12 ? VPI_IMAGE_FORMAT_NV12_BL
                                             : VPI_IMAGE_FORMAT_Y8_ER_BL;
    const int32_t grid = v->gridsize;
    CHECK_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA,
                                        w, h, inFmt,
                                        &grid, 1,
                                        VPI_OPTICAL_FLOW_QUALITY_MEDIUM,
                                        &v->ofPayload));

    uint64_t flags = VPI_BACKEND_CUDA | VPI_BACKEND_OFA;
    CHECK_VPI(vpiImageCreate(w, h, inFmt, flags, &v->curImg));
    CHECK_VPI(vpiImageCreate(w, h, inFmt, flags, &v->prevImg));
    CHECK_VPI(vpiImageCreate(v->outW, v->outH,
                             VPI_IMAGE_FORMAT_2S16_BL,
                             flags, &v->mvImg));

    cudaHostAlloc(&v->flowBufHost, 4 * v->outW * v->outH, cudaHostAllocDefault);
    cudaHostAlloc(&v->costBufHost,     v->outW * v->outH, cudaHostAllocDefault);

    std::memset(v->flowBufHost, 0,   4 * v->outW * v->outH);
    std::memset(v->costBufHost, 255,     v->outW * v->outH);
    v->count = 0;
}

// copy helper
static void copy_image_cuda(image_t *src, VPIImage dst, bool is_nv12, CUstream str)
{
    VPIImageData d{};
    CHECK_VPI(vpiImageLockData(dst, VPI_LOCK_WRITE,
                               VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &d));

    // plane 0 (Y/mono)
    CUDA_MEMCPY2D cp{};
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice     = (CUdeviceptr)src->device_mem;
    cp.dstDevice     = (CUdeviceptr)d.buffer.pitch.planes[0].data;
    cp.srcPitch      = src->stride_y;
    cp.dstPitch      = d.buffer.pitch.planes[0].pitchBytes;
    cp.WidthInBytes  = src->width;
    cp.Height        = src->height;
    cuMemcpy2DAsync(&cp, str);

    if (is_nv12) {
        // plane 1 (UV)
        CUDA_MEMCPY2D cp2{};
        cp2.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cp2.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cp2.srcDevice     = (CUdeviceptr)src->device_mem + src->stride_y * src->height;
        cp2.dstDevice     = (CUdeviceptr)d.buffer.pitch.planes[1].data;
        cp2.srcPitch      = src->stride_uv;
        cp2.dstPitch      = d.buffer.pitch.planes[1].pitchBytes;
        cp2.WidthInBytes  = src->width;
        cp2.Height        = src->height / 2;
        cuMemcpy2DAsync(&cp2, str);
    }

    vpiImageUnlock(dst);
}

// -------------------------------------------------------------------------
// main execute
// -------------------------------------------------------------------------
nvof_results_t *nvof_execute(nvof *v, image_t *img_in)
{
    int newW = v->max_width  & ~3;
    int newH = ((int64_t)newW * img_in->height / img_in->width) & ~3;
    if (newH > v->max_height) {
        newH = v->max_height & ~3;
        newW = ((int64_t)newH * img_in->width / img_in->height) & ~3;
    }
    nvof_set_size(v, newW, newH);

    image_t *scaled = image_scale(img_in, v->width, v->height);
    image_format_t tfmt = v->use_nv12 ? IMAGE_FORMAT_NV12_DEVICE
                                      : IMAGE_FORMAT_MONO_DEVICE;
    image_t *img = image_convert(scaled, tfmt);

    cuda_stream_add_dependency(v->cuStream, img->stream);
    copy_image_cuda(img, v->curImg, v->use_nv12, v->cuStream);

    if (v->count > 0) {
        CHECK_VPI(vpiSubmitOpticalFlowDense(v->vpiStream, 0,
                                            v->ofPayload,
                                            v->prevImg, v->curImg, v->mvImg));
        vpiStreamSync(v->vpiStream);

        VPIImageData mv{};
        CHECK_VPI(vpiImageLockData(v->mvImg, VPI_LOCK_READ,
                                   VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mv));

        const uint8_t *src = (const uint8_t*)mv.buffer.pitch.planes[0].data;
        size_t srcPitch    = mv.buffer.pitch.planes[0].pitchBytes;

        for (int y = 0; y < v->outH; ++y) {
            std::memcpy(v->flowBufHost + y * v->outW,
                        src + y * srcPitch,
                        v->outW * sizeof(flow_vector_t));
        }
        vpiImageUnlock(v->mvImg);
        std::memset(v->costBufHost, 255, v->outW * v->outH);
    } else {
        std::memset(v->flowBufHost, 0,   4 * v->outW * v->outH);
        std::memset(v->costBufHost, 255,     v->outW * v->outH);
        vpiStreamSync(v->vpiStream);
    }

    ++v->count;
    std::swap(v->curImg, v->prevImg);

    v->results.grid_w = v->outW;
    v->results.grid_h = v->outH;
    v->results.flow   = v->flowBufHost;
    v->results.costs  = v->costBufHost;

    image_destroy(img);
    image_destroy(scaled);
    return &v->results;
}

// quick “no-motion” helper
void nvof_set_no_motion(nvof *v)
{
    if (!v) return;
    std::memset(v->flowBufHost, 0,   4 * v->outW * v->outH);
    std::memset(v->costBufHost, 0,       v->outW * v->outH);
}

// -------------------------------------------------------------------------
// ctor / dtor
// -------------------------------------------------------------------------
nvof *nvof_create(void*, int max_w, int max_h)
{
    check_cuda_inited();

    nvof *n = (nvof*)std::calloc(1, sizeof(nvof));
    if (!n) return nullptr;

    n->max_width  = max_w;
    n->max_height = max_h;
    n->use_nv12   = true;

    n->cuStream = create_cuda_stream();
    CHECK_VPI(vpiStreamCreateWrapperCUDA(n->cuStream,
                                         VPI_BACKEND_CUDA | VPI_BACKEND_OFA,
                                         &n->vpiStream));
    return n;
}

void nvof_destroy(nvof *n)
{
    if (!n) return;
    vpiStreamDestroy(n->vpiStream);
    destroy_host_buffers(n);
    destroy_vpi_image(n->curImg);
    destroy_vpi_image(n->prevImg);
    destroy_vpi_image(n->mvImg);
    if (n->ofPayload) vpiPayloadDestroy(n->ofPayload);
    destroy_cuda_stream(n->cuStream);
    std::free(n);
}

#endif