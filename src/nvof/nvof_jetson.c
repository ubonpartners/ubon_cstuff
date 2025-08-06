// nvof_vpi.cpp – Jetson Orin Nano implementation of the NVOF interface using VPI Dense Optical Flow

#if (UBONCSTUFF_PLATFORM == 1)
// nvof_jetson.c  –  Jetson Orin Nano implementation using VPI-3 Dense OF
#include "nvof.h"
#include "log.h"
#include "cuda_stuff.h"
#include "display.h"

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/CUDAInterop.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/OpticalFlowDense.h>

#include <cstring>
#include <mutex>

#define CHECK_VPI(call)                                                     \
    do {                                                                    \
        VPIStatus _st = (call);                                             \
        if (_st != VPI_SUCCESS) {                                           \
            char msg[VPI_MAX_STATUS_MESSAGE_LENGTH];                        \
            vpiGetLastStatusMessage(msg, sizeof(msg));                      \
            log_fatal("%s failed: %s (%d)", #call, msg, _st);               \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* --------------------------------------------------------------------- */
/* Internal structure – mirrors desktop version but follows sample flow */
/* --------------------------------------------------------------------- */
struct nvof
{
    /* limits */
    int max_w = 0, max_h = 0;

    /* current working resolution */
    int w = 0, h = 0;
    int grid = 4, outW = 0, outH = 0;

    /* CUDA + VPI */
    CUstream  cu;        // created with create_cuda_stream()
    VPIStream vpi;       // wrapper around |cu|

    /* VPI resources --------------------------------------------------- */
    VPIImage  curNv12PL  = nullptr;   // wrapper (updated each frame)
    
    /* NV12 working buffers */
    VPIImage  nv12CurBL  = nullptr;   // block‑linear NV12  (VIC/OFA)
    VPIImage  nv12PrevBL = nullptr;

    VPIImage  mvBL       = nullptr;   // block-linear 2S16
    VPIImage  mvPL       = nullptr;   // pitch-linear 2S16 (CPU read-back)

    VPIPayload ofPayload = nullptr;

    /* host buffers returned to caller -------------------------------- */
    flow_vector_t *flowHost = nullptr;
    /* misc */
    uint32_t frameCount = 0;
    bool     use_nv12   = true;
    bool     nv12WrapInit = false;

    nvof_results_t results{};
};

/* --------------------------------------------------------------------- */
/* helpers                                                              */
/* --------------------------------------------------------------------- */
static void destroyVPI(VPIImage &img)        { if (img) vpiImageDestroy(img), img=nullptr; }
static void destroyBuf(nvof *v)
{
    if (v->flowHost) cudaFreeHost(v->flowHost), v->flowHost=nullptr;
}

/* (re)alloc everything whenever resolution changes ------------------- */
static void set_size(nvof *v, int w, int h)
{
    if (v->w == w && v->h == h) return;

    destroyVPI(v->nv12CurBL); destroyVPI(v->nv12PrevBL);
    destroyVPI(v->mvBL);     destroyVPI(v->mvPL);
    if (v->ofPayload) { vpiPayloadDestroy(v->ofPayload); v->ofPayload=nullptr; }
    destroyBuf(v);
    /* note: curNv12PL wrapper is kept – pointer is updated each frame */

    v->w=w; v->h=h; v->grid=4;
    v->outW=(w+3)/4; v->outH=(h+3)/4;

    uint64_t cuda  = VPI_BACKEND_CUDA;
    uint64_t vic   = VPI_BACKEND_VIC;
    uint64_t ofa   = VPI_BACKEND_OFA;

    /* block‑linear NV12 images (no CUDA backend needed) */
    uint64_t blFlags = VPI_BACKEND_VIC | VPI_BACKEND_OFA;
    CHECK_VPI(vpiImageCreate(w,h,VPI_IMAGE_FORMAT_NV12_BL,blFlags,&v->nv12CurBL));
    CHECK_VPI(vpiImageCreate(w,h,VPI_IMAGE_FORMAT_NV12_BL,blFlags,&v->nv12PrevBL));

    /* motion-vector images */
    CHECK_VPI(vpiImageCreate(v->outW,v->outH,VPI_IMAGE_FORMAT_2S16_BL,vic|ofa,&v->mvBL));
    CHECK_VPI(vpiImageCreate(v->outW,v->outH,VPI_IMAGE_FORMAT_2S16,0,&v->mvPL));

    /* allocate OFA payload (single level) */
    int32_t g = v->grid;
    CHECK_VPI(vpiCreateOpticalFlowDense(ofa, w, h,
                                        VPI_IMAGE_FORMAT_NV12_BL,
                                        &g, 1,
                                        VPI_OPTICAL_FLOW_QUALITY_MEDIUM,
                                        &v->ofPayload));

    VPIOpticalFlowDenseSGMParams sgm{};
    vpiOpticalFlowDenseGetSGMParams(v->ofPayload, &sgm);

    // e.g., make flow smoother by raising both penalties on level 0:
    sgm.p1[0]      = 8;    // default is ~4
    sgm.p2[0]      = 64;   // default is ~32
    sgm.p2Alpha[0] = 4;    // turn on adaptive P2
    sgm.numPasses[0]      = 2;
    sgm.includeDiagonals[0] = 1;

    CHECK_VPI(vpiOpticalFlowDenseSetSGMParams(v->ofPayload, &sgm));

    /* host pinned result buffers */
    cudaHostAlloc(&v->flowHost, 4*v->outW*v->outH, cudaHostAllocDefault);
    memset(v->flowHost,0,4*v->outW*v->outH);

    v->frameCount   = 0;
    v->nv12WrapInit = false;
}

static void update_nv12_wrapper(nvof *v, image_t *img)
{
    /* Describe current buffer --------------------------------------- */
    VPIImageData d{};
    d.bufferType               = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    d.buffer.pitch.format      = VPI_IMAGE_FORMAT_NV12;
    d.buffer.pitch.numPlanes   = 2;

    d.buffer.pitch.planes[0].width      = img->width;
    d.buffer.pitch.planes[0].height     = img->height;
    d.buffer.pitch.planes[0].pitchBytes = img->stride_y;
    d.buffer.pitch.planes[0].data       = img->y;

    d.buffer.pitch.planes[1].width      = img->width / 2;
    d.buffer.pitch.planes[1].height     = img->height / 2;
    d.buffer.pitch.planes[1].pitchBytes = img->stride_uv;
    d.buffer.pitch.planes[1].data       = img->u;

    /* Throw away the old wrapper (if any) ---------------------------- */
    if (v->curNv12PL)
        vpiImageDestroy(v->curNv12PL), v->curNv12PL = nullptr;

    /* Re‑create with the back‑ends that will touch it (CUDA+VIC) ----- */
    VPIImageWrapperParams p;
    vpiInitImageWrapperParams(&p);
    p.colorSpec = VPI_COLOR_SPEC_BT601_ER;      // same as before

    CHECK_VPI(vpiImageCreateWrapper(&d, &p,
                                    VPI_BACKEND_CUDA | VPI_BACKEND_VIC,
                                    &v->curNv12PL));
}


/* copy MV 2S16 image into caller-visible buffers ---------------------- */
static void export_motion_vectors(nvof *v)
{
    VPIImageData mv{};
    CHECK_VPI(vpiImageLockData(v->mvPL, VPI_LOCK_READ,
                               VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mv));
    auto src =(const uint8_t*)mv.buffer.pitch.planes[0].data;
    size_t sp = mv.buffer.pitch.planes[0].pitchBytes;
    for(int y=0;y<v->outH;++y)
        memcpy(v->flowHost + y*v->outW, src + y*sp, v->outW*sizeof(flow_vector_t));
    vpiImageUnlock(v->mvPL);
}

/* --------------------------------------------------------------------- */
/* public API                                                            */
/* --------------------------------------------------------------------- */
nvof_results_t *nvof_execute(nvof *v, image_t *img_in)
{
    /* decide working resolution like the desktop path ---------------- */
    int w=v->max_w &~3;
    int h=((int64_t)w*img_in->height/img_in->width)&~3;
    if(h>v->max_h){ h=v->max_h&~3; w=((int64_t)h*img_in->width/img_in->height)&~3; }
    set_size(v,w,h);

    /* scale/convert to NV12 device buffer (we reuse existing helpers) -- */
    image_t *scaled = image_scale(img_in, v->w, v->h);
    image_format_t tf = IMAGE_FORMAT_NV12_DEVICE;
    image_t *img = image_convert(scaled, tf);

    /* update NV12 wrapper to point to this frame --------------------- */
    cuda_stream_add_dependency(v->cu, img->stream);
    update_nv12_wrapper(v,img);

    /* NV12_PL -> NV12_BL (VIC) -------------------------------------- */
    CHECK_VPI(vpiSubmitConvertImageFormat(v->vpi, VPI_BACKEND_VIC,
                                          v->curNv12PL, v->nv12CurBL, nullptr));

    if (v->frameCount>0) {
        /* run OFA ---------------------------------------------------- */
        CHECK_VPI(vpiSubmitOpticalFlowDense(v->vpi, 0,
                                            v->ofPayload,
                                            v->nv12PrevBL, v->nv12CurBL,
                                            v->mvBL));

        /* MV BL -> PL (VIC) ----------------------------------------- */
        CHECK_VPI(vpiSubmitConvertImageFormat(v->vpi, VPI_BACKEND_VIC,
                                              v->mvBL, v->mvPL, nullptr));
        CHECK_VPI(vpiStreamSync(v->vpi));
        export_motion_vectors(v);
    } else {
        memset(v->flowHost,0,4*v->outW*v->outH);
        CHECK_VPI(vpiStreamSync(v->vpi));
    }

    ++v->frameCount;

    /* swap prev/cur -------------------------------------------------- */
    std::swap(v->nv12CurBL , v->nv12PrevBL );

    /* results -------------------------------------------------------- */
    v->results.grid_w = v->outW;
    v->results.grid_h = v->outH;
    v->results.flow   = v->flowHost;
    v->results.costs  = 0;

    image_destroy(img); image_destroy(scaled);
    return &v->results;
}

/* --------------------------------------------------------------------- */
nvof *nvof_create(void*, int max_w, int max_h)
{
    check_cuda_inited();

    nvof *n=(nvof*)calloc(1,sizeof(nvof));
    n->max_w=max_w; n->max_h=max_h;

    n->cu = create_cuda_stream();
    CHECK_VPI(vpiStreamCreateWrapperCUDA(n->cu,
                                         VPI_BACKEND_CUDA|
                                         VPI_BACKEND_VIC |
                                         VPI_BACKEND_OFA, &n->vpi));
    return n;
}

void nvof_destroy(nvof *n)
{
    if(!n) return;
    vpiStreamDestroy(n->vpi);
    destroyVPI(n->curNv12PL);
    destroyVPI(n->nv12CurBL); destroyVPI(n->nv12PrevBL);
    destroyVPI(n->mvBL);     destroyVPI(n->mvPL);
    if (n->ofPayload) vpiPayloadDestroy(n->ofPayload);
    destroyBuf(n);
    destroy_cuda_stream(n->cu);
    free(n);
}

void nvof_set_no_motion(nvof *v)
{
    if(!v) return;
    memset(v->flowHost,0,4*v->outW*v->outH);
}

void nvof_reset(nvof_t *n)
{
    if (!n) return;
    n->frameCount=0;
}

#endif /* UBONCSTUFF_PLATFORM == 1 */
