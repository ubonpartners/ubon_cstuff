#include <npp.h>
#include <cuda.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "image.h"
#include "libyuv.h"
#include "cuda_stuff.h"
#include "cuda_kernels.h"
#include "misc.h"
#include <mutex>

#define CHECK_NPPcall(call) \
    do { \
        NppStatus _status = (call); \
        if (_status != NPP_SUCCESS) { \
            log_fatal("NPP error code %d", _status); \
            exit(1); \
        } \
    } while (0)

static image_t *image_scale_yuv420_host(image_t *src, int width, int height)
{
    image_t *dst=create_image(width, height, IMAGE_FORMAT_YUV420_HOST);
    if (!dst) return 0;
    image_add_dependency(dst, src); 
    I420Scale(
            src->y, src->stride_y,
            src->u, src->stride_uv,
            src->v, src->stride_uv,
            src->width, src->height,
            dst->y, dst->stride_y,
            dst->u, dst->stride_uv,
            dst->v, dst->stride_uv,
            dst->width, dst->height,
            libyuv::kFilterBilinear
            );
    return dst;
}

static image_t *image_scale_yuv420_device(image_t *src, int width, int height)
{
    if (0)//src->width>2*width && src->height>2*height)
    {
        if (((src->width&3)==0) && ((src->height&3)==0))
        {
            int inter_w=src->width>>1;
            int inter_h=src->height>>1;
            image_t *inter=create_image(inter_w, inter_h, IMAGE_FORMAT_YUV420_DEVICE);
            if (!inter) return 0;
            image_add_dependency(inter, src); // don't run this until 'src' is ready
            cuda_downsample_2x2(src->y, src->stride_y, inter->y, inter->stride_y, inter_w, inter_w, inter->stream);
            cuda_downsample_2x2(src->u, src->stride_uv, inter->u, inter->stride_uv, inter_w/2, inter_h/2, inter->stream);
            cuda_downsample_2x2(src->v, src->stride_uv, inter->v, inter->stride_uv, inter_w/2, inter_h/2, inter->stream);
            image_t *ret=image_scale_yuv420_device(inter, width, height);
            destroy_image(inter);
            return ret;
        }
    }

    image_t *dst=create_image(width, height, IMAGE_FORMAT_YUV420_DEVICE);
    if (!dst) return 0;
    image_add_dependency(dst, src); // don't run this until 'src' is ready

    NppiSize srcSize = {src->width, src->height};
    NppiRect srcROI = {0, 0, src->width, src->height};
    NppiRect dstROI = {0, 0, dst->width, dst->height};

    NppStreamContext nppStreamCtx=get_nppStreamCtx();
    nppStreamCtx.hStream=dst->stream;

    // Create scaling context for Y plane
    NppiInterpolationMode interpolationMode = NPPI_INTER_LINEAR;
    NppiSize dstSize = {dst->width, dst->height};

    // Y plane scaling
    CHECK_NPPcall(nppiResize_8u_C1R_Ctx(src->y, src->stride_y, srcSize, srcROI,
                                         dst->y, dst->stride_y, dstSize, dstROI,
                                         interpolationMode, nppStreamCtx));

    // Scale U and V planes (quarter resolution of Y in YUV420)
    NppiSize srcSizeUV = {src->width / 2, src->height / 2};
    NppiRect srcROIUV = {0, 0, src->width / 2, src->height / 2};
    NppiSize dstSizeUV = {dst->width / 2, dst->height / 2};
    NppiRect dstROIUV = {0, 0, dst->width / 2, dst->height / 2};

    // U plane scaling
    CHECK_NPPcall(nppiResize_8u_C1R_Ctx(src->u, src->stride_uv, srcSizeUV, srcROIUV,
                               dst->u, dst->stride_uv, dstSizeUV, dstROIUV,
                               interpolationMode, nppStreamCtx));

    // V plane scaling
    CHECK_NPPcall(nppiResize_8u_C1R_Ctx(src->v, src->stride_uv, srcSizeUV, srcROIUV,
                               dst->v, dst->stride_uv, dstSizeUV, dstROIUV,
                               interpolationMode, nppStreamCtx));
    return dst;
}

static image_t *image_scale_mono_device(image_t *src, int width, int height)
{
    if (src->width>2*width && src->height>2*height)
    {
        if (((src->width&1)==0) && ((src->height&1)==0))
        {
            int inter_w=src->width>>1;
            int inter_h=src->height>>1;
            image_t *inter=create_image(inter_w, inter_h, IMAGE_FORMAT_MONO_DEVICE);
            if (!inter) return 0;
            image_add_dependency(inter, src); // don't run this until 'src' is ready
            cuda_downsample_2x2(src->y, src->stride_y, inter->y, inter->stride_y, inter_w, inter_w, inter->stream);
            image_t *ret=image_scale_mono_device(inter, width, height);
            destroy_image(inter);
            return ret;
        }
    }

    image_t *dst=create_image(width, height, IMAGE_FORMAT_MONO_DEVICE);
    if (!dst) return 0;
    image_add_dependency(dst, src); // don't run this until 'src' is ready

    NppiSize srcSize = {src->width, src->height};
    NppiRect srcROI = {0, 0, src->width, src->height};
    NppiRect dstROI = {0, 0, dst->width, dst->height};

    NppStreamContext nppStreamCtx=get_nppStreamCtx();
    nppStreamCtx.hStream=dst->stream;

    // Create scaling context for Y plane
    NppiInterpolationMode interpolationMode = NPPI_INTER_LINEAR;
    NppiSize dstSize = {dst->width, dst->height};

    // Y plane scaling
    CHECK_NPPcall(nppiResize_8u_C1R_Ctx(src->y, src->stride_y, srcSize, srcROI,
                                         dst->y, dst->stride_y, dstSize, dstROI,
                                         interpolationMode, nppStreamCtx));
    return dst;
}

static image_t *image_scale_by_intermediate(image_t *img, int width, int height, image_format_t inter)
{
    assert(img->format!=inter);
    image_t *image_tmp=image_convert(img, inter);
    assert(image_tmp!=0);
    image_t *scaled=image_scale(image_tmp, width, height);
    assert(scaled!=0);
    destroy_image(image_tmp);
    return scaled;
}

image_t *image_scale(image_t *img, int width, int height)
{
    if (img->width==width && img->height==height)
    {
        return image_reference(img);
    }

    switch(img->format)
    {
        case IMAGE_FORMAT_YUV420_HOST:
            return image_scale_yuv420_host(img,width,height);
        case IMAGE_FORMAT_YUV420_DEVICE:
            return image_scale_yuv420_device(img,width,height);
        case IMAGE_FORMAT_NV12_DEVICE:
            return image_scale_by_intermediate(img,width,height,IMAGE_FORMAT_YUV420_DEVICE);
        case IMAGE_FORMAT_RGB24_HOST:
            return image_scale_by_intermediate(img,width,height,IMAGE_FORMAT_YUV420_DEVICE);
        case IMAGE_FORMAT_MONO_DEVICE:
            return image_scale_mono_device(img,width,height);
        default:
        ;
    }
    return 0;
}