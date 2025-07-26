/**********************************************************************
 *  bl2pl_copy.cu   (Jetson ‑ JP 4.6/5.x/6.x)
 *
 *  Copy one decoder NvBufSurface (Block‑Linear YUV420M) ➜ pitch‑linear
 *  YUV420 image_t already resident in CUDA device memory.
 *
 *  • zero host copies, no NvTransform()
 *  • one device memcpy per plane (BL ➜ PL swizzle done by driver)
 *********************************************************************/

 #if UBONCSTUFF_PLATFORM == 1

#include <cuda.h>          // driver API
#include <cudaEGL.h>
#include <EGL/egl.h>

#include "NvBufSurface.h"
#include "image.h"         // your image_t  { y,u,v, stride_y/uv, width, height }

/* ------------------------------------------------------------------ */
/*  RAII helpers                                                      */
/* ------------------------------------------------------------------ */
struct CuResRAII {
    CUgraphicsResource res{nullptr};
    ~CuResRAII() { if (res) cuGraphicsUnregisterResource(res); }
};

struct SurfEglMap {
    NvBufSurface *surf;
    int idx;
    SurfEglMap(NvBufSurface *s, int i) : surf(s), idx(i)  { NvBufSurfaceMapEglImage(surf, idx); }
    ~SurfEglMap()                                        { NvBufSurfaceUnMapEglImage(surf, idx); }
};

/* ------------------------------------------------------------------ */
/*  copy ONE plane  (Block‑Linear ➜ pitch‑linear)                      */
/* ------------------------------------------------------------------ */
static inline CUresult copyPlaneBLtoPL(NvBufSurface *nvSurf,
                                       int           plane,
                                       uint8_t      *dstPtr,
                                       size_t        dstPitch,
                                       int           widthBytes,
                                       int           height,
                                       CUstream      stream = 0)
{
    /* 1. Map the surface –> EGLImage (all planes) */
    SurfEglMap map{nvSurf, 0};

    /* Since JP‑4.6 the field is a single void*; just cast it.              */
    EGLImageKHR eglImg = static_cast<EGLImageKHR>(nvSurf->surfaceList[0].mappedAddr.eglImage);

    /* 2. Register with CUDA driver                                         */
    CuResRAII       cuRes;
    CUresult        cuErr;
    cuErr = cuGraphicsEGLRegisterImage(&cuRes.res, eglImg,
                                       CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
    if (cuErr != CUDA_SUCCESS)
    {
        printf("cuGraphicsEGLRegisterImage failed error %d\n", (int)cuErr);
        return cuErr;
    }
    /* 3. Obtain a CUarray for the requested plane                          */
    CUeglFrame eglFrame;
    cuErr = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cuRes.res, 0, 0);
    if (cuErr != CUDA_SUCCESS)
    {
        printf("cuGraphicsResourceGetMappedEglFrame failed error %d", (int)cuErr);
        return cuErr;
    }
    CUarray cuArr = eglFrame.frame.pArray[plane];          // per‑plane array

    /* 4. BL ➜ PL copy (driver de‑swizzle)                                 */
    CUDA_MEMCPY2D cp = {};
    cp.srcMemoryType  = CU_MEMORYTYPE_ARRAY;
    cp.srcArray       = cuArr;
    cp.dstMemoryType  = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice      = reinterpret_cast<CUdeviceptr>(dstPtr);
    cp.dstPitch       = dstPitch;
    cp.WidthInBytes   = static_cast<size_t>(widthBytes);
    cp.Height         = static_cast<size_t>(height);

    return cuMemcpy2DAsync(&cp, stream);
}

/* ------------------------------------------------------------------ */
/*  Public helper                                                     */
/* ------------------------------------------------------------------ */
/**
 * nvSurfToImageYUV420Device
 *
 * Copy one NvBufSurface (decoder, BL YUV420M) into a pitch‑linear
 * YUV420 `image_t` that lives in CUDA device memory.
 *
 * @return 0 on success, -1 on failure
 */

static inline bool is_nv12_variant(NvBufSurfaceColorFormat fmt)
{
    switch (fmt) {
        case NVBUF_COLOR_FORMAT_NV12:
        case NVBUF_COLOR_FORMAT_NV12_ER:
        case NVBUF_COLOR_FORMAT_NV12_709:
        case NVBUF_COLOR_FORMAT_NV12_709_ER:
        case NVBUF_COLOR_FORMAT_NV12_2020:       
            return true;
        default:
            return false;
    }
}

int nvSurfToImageNV12Device(NvBufSurface *nvSurf,
                              image_t      *img,
                              CUstream      stream)
{
    if (!nvSurf || !img)                                                return -1;
    if (!is_nv12_variant(nvSurf->surfaceList[0].colorFormat))
    {
        printf("Bad colour format %d\n",(int)nvSurf->surfaceList[0].colorFormat);
        return -1;
    }
    const int W    = img->width;
    const int H    = img->height;

    if (copyPlaneBLtoPL(nvSurf, 0, img->y, img->stride_y,  W,   H,   stream) != CUDA_SUCCESS) return -1;
    if (copyPlaneBLtoPL(nvSurf, 1, img->u, img->stride_uv, W, H/2, stream) != CUDA_SUCCESS) return -1;

    return 0;
}

/* ---------- Example use inside your decode loop --------------------

NvBufSurface *surf = nullptr;
NvBufSurfaceFromFd(dec_buffer->planes[0].fd, (void**)&surf);

image_t *img = create_image(w, h, IMAGE_FORMAT_YUV420_DEVICE);
nvSurfToImageYUV420Device(surf, img, 0);   // 0 = default CUstream

------------------------------------------------------------------------*/
#endif
