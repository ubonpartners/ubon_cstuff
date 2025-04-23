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

void clear_image(image_t *img)
{
    if (!img) return;
    if (img->device_mem_size>0) cudaMemsetAsync((void*)img->device_mem, 0, img->device_mem_size, img->stream);
    if (img->host_mem_size>0)
    {
        image_sync(img);
        memset(img->host_mem, 0, img->host_mem_size);
    }
}

uint32_t image_hash(image_t *img)
{
    if (img==0) return 0;
    if (img->host_mem!=0)
        return hash_host(img->host_mem, img->host_mem_size);
    else 
        return hash_gpu((void*)img->device_mem, img->device_mem_size, img->stream);
}

image_t *image_blur(image_t *img) 
{
    if (!img || (img->format != IMAGE_FORMAT_YUV420_DEVICE && img->format != IMAGE_FORMAT_MONO_DEVICE)) 
    {
        return NULL;
    }

    // Create output image of the same format and size
    image_t *dst = create_image(img->width, img->height, img->format);
    image_add_dependency(dst, img); // ensure dst processing waits for img
    NppStreamContext nppStreamCtx = get_nppStreamCtx();
    nppStreamCtx.hStream = dst->stream;

    NppiSize size = { img->width, img->height };
    NppiMaskSize mask = NPP_MASK_SIZE_5_X_5;
    NppiBorderType border = NPP_BORDER_REPLICATE;
    NppiPoint offset = { 0, 0 };

    // Blur Y channel
    CHECK_NPPcall(nppiFilterGaussBorder_8u_C1R_Ctx(
        img->y, img->stride_y,
        size, offset,
        dst->y, dst->stride_y,
        size, mask, border, nppStreamCtx));

    if (img->format == IMAGE_FORMAT_YUV420_DEVICE) {
        NppiSize uv_size = { img->width / 2, img->height / 2 };

        // Blur U channel
        CHECK_NPPcall(nppiFilterGaussBorder_8u_C1R_Ctx(
            img->u, img->stride_uv,
            uv_size, offset,
            dst->u, dst->stride_uv,
            uv_size, mask, border, nppStreamCtx));

        // Blur V channel
        CHECK_NPPcall(nppiFilterGaussBorder_8u_C1R_Ctx(
            img->v, img->stride_uv,
            uv_size, offset,
            dst->v, dst->stride_uv,
            uv_size, mask, border, nppStreamCtx));
    }

    return dst;
}

image_t *image_mad_4x4(image_t *a, image_t *b) 
{
    if (!a || !b) return NULL;
    if (a->width != b->width || a->height != b->height)
        return NULL;
    if (a->format!=b->format) return NULL;

    if ((a->format!=IMAGE_FORMAT_MONO_DEVICE)&&(a->format!=IMAGE_FORMAT_YUV420_DEVICE))
    {
        image_t *inter_a=image_convert(a, IMAGE_FORMAT_YUV420_DEVICE);
        image_t *inter_b=image_convert(b, IMAGE_FORMAT_YUV420_DEVICE);
        image_t *ret=image_mad_4x4(inter_a, inter_b);
        destroy_image(inter_a);
        destroy_image(inter_b);
        return ret;
    }

    int out_width = a->width / 4;
    int out_height = a->height / 4;
    image_t *out = create_image(out_width, out_height, a->format);
    image_add_dependency(out, a);
    image_add_dependency(out, b);
    compute_4x4_mad_mask(a->y, a->stride_y, b->y, b->stride_y,
        out->y, out->stride_y, out_width, out_height, out->stream);

    if (a->format==IMAGE_FORMAT_YUV420_DEVICE)
    {
        compute_4x4_mad_mask(a->u, a->stride_uv, b->u, b->stride_uv,
            out->u, out->stride_uv, out_width>>1, out_height>>1, out->stream);
        compute_4x4_mad_mask(a->v, a->stride_uv, b->v, b->stride_uv,
            out->v, out->stride_uv, out_width>>1, out_height>>1, out->stream);
    }

    return out;
}