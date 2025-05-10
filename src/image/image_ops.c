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

uint32_t hash_plane(uint8_t *p, int w, int h, int stride, bool device, cudaStream_t stream)
{
    uint32_t hashes[h];
    if (device==false)
    {
        CHECK_CUDART_CALL(cudaStreamSynchronize(stream));
        hash_2d(p, w, h, stride, hashes);
    }
    else
    {
        cuda_hash_2d(p, w, h, stride, hashes, stream);
        CHECK_CUDART_CALL(cudaStreamSynchronize(stream));
    }
    return hash_u32(hashes, h);
}

uint32_t image_hash(image_t *img)
{
    if (img==0) return 0;
    uint32_t hashes[4];
    int nhashes=0;
    bool is_device=image_format_is_device(img->format);

    if (  (img->format==IMAGE_FORMAT_YUV420_DEVICE)
        ||(img->format==IMAGE_FORMAT_MONO_DEVICE)
        ||(img->format==IMAGE_FORMAT_YUV420_HOST)
        ||(img->format==IMAGE_FORMAT_MONO_HOST)
        ||(img->format==IMAGE_FORMAT_NV12_DEVICE))
        hashes[nhashes++]=hash_plane(img->y, img->width, img->height, img->stride_y, is_device, img->stream);
    if ((img->format==IMAGE_FORMAT_YUV420_DEVICE)||(img->format==IMAGE_FORMAT_YUV420_HOST))
    {
        hashes[nhashes++]=hash_plane(img->u, img->width/2, img->height/2, img->stride_uv, is_device, img->stream);
        hashes[nhashes++]=hash_plane(img->v, img->width/2, img->height/2, img->stride_uv, is_device, img->stream);
    }
    if (img->format==IMAGE_FORMAT_NV12_DEVICE)
    {
        hashes[nhashes++]=hash_plane(img->u, img->width, img->height/2, img->stride_uv, is_device, img->stream);
    }
    if ((img->format==IMAGE_FORMAT_RGB24_DEVICE)||(img->format==IMAGE_FORMAT_RGB24_HOST))
    {
        hashes[nhashes++]=hash_plane(img->rgb, img->width*3, img->height, img->stride_rgb, is_device, img->stream);
    }
    if ((img->format==IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE)||(img->format==IMAGE_FORMAT_RGB_PLANAR_FP32_HOST))
    {
        hashes[nhashes++]=hash_plane(img->rgb, img->width*3*4, img->height, img->stride_rgb, is_device, img->stream);
    }
    if ((img->format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE)||(img->format==IMAGE_FORMAT_RGB_PLANAR_FP16_HOST))
    {
        hashes[nhashes++]=hash_plane(img->rgb, img->width*3*2, img->height, img->stride_rgb, is_device, img->stream);
    }
    uint32_t ret=hash_u32(hashes, nhashes);
    return ret;
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

image_t *image_blend(image_t *src, image_t *src2, int sx, int sy, int w, int h, int dx, int dy)
{
    if (!src) return 0;
    if (!src2) return image_reference(src);

    if ((src->format != IMAGE_FORMAT_YUV420_DEVICE && src->format != IMAGE_FORMAT_MONO_DEVICE))
    {
        image_t *inter=image_convert(src, IMAGE_FORMAT_YUV420_DEVICE);
        image_t *ret=image_blend(inter, src2, sx, sy, w, h, dx, dy);
        destroy_image(inter);
        return ret;
    }

    if (src2->format!=src->format)
    {
        image_t *inter=image_convert(src2, src->format);
        image_t *ret=image_blend(src, inter, sx, sy, w, h, dx, dy);
        destroy_image(inter);
        return ret;
    }

    // check bounds
    w=std::max(0, std::min(std::min(w, src2->width-sx), src->width-dx));
    h=std::max(0, std::min(std::min(h, src2->height-sy), src->height-dy));
    if (w==0 || h==0) return image_reference(src);

    // Create a copy of src to modify and return
    image_t *out = create_image(src->width, src->height, src->format);
    image_add_dependency(out, src);
    image_add_dependency(out, src2);

    // Copy full Y plane from src into out
    CHECK_CUDART_CALL(cudaMemcpy2DAsync(out->y, out->stride_y,
                      src->y, src->stride_y,
                      src->width, src->height,
                      cudaMemcpyDeviceToDevice,
                      out->stream));

    // Copy selected region from src2 Y plane
    for (int row = 0; row < h; row++) {
        CHECK_CUDART_CALL(cudaMemcpyAsync(
            out->y + (dy + row) * out->stride_y + dx,
            src2->y + (sy + row) * src2->stride_y + sx,
            w,
            cudaMemcpyDeviceToDevice,
            out->stream));
    }

    if (src->format == IMAGE_FORMAT_YUV420_DEVICE) {
        int uv_w = w / 2;
        int uv_h = h / 2;
        int sx_uv = sx / 2, sy_uv = sy / 2;
        int dx_uv = dx / 2, dy_uv = dy / 2;

        // Copy full U and V planes from src into out
        CHECK_CUDART_CALL(cudaMemcpy2DAsync(out->u, out->stride_uv, src->u, src->stride_uv,
                          src->width / 2, src->height / 2,
                          cudaMemcpyDeviceToDevice, out->stream));

        CHECK_CUDART_CALL(cudaMemcpy2DAsync(out->v, out->stride_uv, src->v, src->stride_uv,
                          src->width / 2, src->height / 2,
                          cudaMemcpyDeviceToDevice, out->stream));

        // Copy U and V blocks
        for (int row = 0; row < uv_h; row++) {
            CHECK_CUDART_CALL(cudaMemcpyAsync(
                out->u + (dy_uv + row) * out->stride_uv + dx_uv,
                src2->u + (sy_uv + row) * src2->stride_uv + sx_uv,
                uv_w,
                cudaMemcpyDeviceToDevice,
                out->stream));

            CHECK_CUDART_CALL(cudaMemcpyAsync(
                out->v + (dy_uv + row) * out->stride_uv + dx_uv,
                src2->v + (sy_uv + row) * src2->stride_uv + sx_uv,
                uv_w,
                cudaMemcpyDeviceToDevice,
                out->stream));
        }
    }

    return out;
}

image_t *image_crop(image_t *img, int x, int y, int w, int h)
{
    if (!img) return 0;
    if ((img->format != IMAGE_FORMAT_YUV420_DEVICE && img->format != IMAGE_FORMAT_MONO_DEVICE))
    {
        image_t *inter=image_convert(img, IMAGE_FORMAT_YUV420_DEVICE);
        image_t *ret=image_crop(inter, x, y, w, h);
        destroy_image(inter);
        return ret;
    }

    w=std::max(0, std::min(w, img->width-x));
    h=std::max(0, std::min(h, img->height-y));

    if (w==0 || h==0) return 0;

    // what we do here is not actually any work - we just create a new shell surface
    // that points to the Y data of the original surface, and hold on to a reference
    // for that surface.
    image_t *cropped=create_image_no_surface_memory(w, h, img->format);
    cropped->referenced_surface=image_reference(img);
    cropped->y=img->y+x+y*img->stride_y;
    cropped->stride_y=img->stride_y;
    if (cropped->format==IMAGE_FORMAT_YUV420_DEVICE)
    {
        cropped->stride_uv=img->stride_uv;
        cropped->u=img->u+(x>>1)+(y>>1)*cropped->stride_uv;
        cropped->v=img->u+(x>>1)+(y>>1)*cropped->stride_uv;
    }
    image_add_dependency(cropped, img);
    return cropped;
}

image_t *image_pad_rgb24_device(image_t *img, int left, int top, int right, int bottom, uint32_t RGB)
{
    int new_width = img->width + left + right;
    int new_height = img->height + top + bottom;

    image_t *dst = create_image(new_width, new_height, img->format);
    if (!dst) return 0;
    image_add_dependency(dst, img); // Wait until img is ready

    // Fill destination with RGB color
    Npp8u rgb_color[3] = {
        (Npp8u)((RGB >> 16) & 0xFF),
        (Npp8u)((RGB >> 8) & 0xFF),
        (Npp8u)(RGB & 0xFF)
    };

    NppStreamContext nppStreamCtx = get_nppStreamCtx();
    nppStreamCtx.hStream = dst->stream;

    typedef struct {int x, y, width, height;} PadRect;

    PadRect pad_rects[4] = {
        { 0, 0, dst->width, top },                                // Top
        { 0, dst->height - bottom, dst->width, bottom },          // Bottom
        { 0, top, left, img->height },                            // Left
        { dst->width - right, top, right, img->height }           // Right
    };

    for (int i = 0; i < 4; ++i) {
        PadRect *r = &pad_rects[i];
        if (r->width > 0 && r->height > 0) {
            NppiSize roiSize = { r->width, r->height };
            Npp8u *ptr = dst->rgb + r->y * dst->stride_rgb + r->x * 3;
            CHECK_NPPcall(nppiSet_8u_C3R_Ctx(rgb_color, ptr, dst->stride_rgb, roiSize, nppStreamCtx));
        }
    }

    // Copy src into padded location in dst
    NppiSize srcSize = { img->width, img->height };
    Npp8u *dstOffsetPtr = dst->rgb + top * dst->stride_rgb + left * 3;

    CHECK_NPPcall(nppiCopy_8u_C3R_Ctx(
        img->rgb, img->stride_rgb,
        dstOffsetPtr, dst->stride_rgb,
        srcSize, nppStreamCtx));

    return dst;
}

image_t *image_pad(image_t *img, int left, int top, int right, int bottom, uint32_t RGB)
{
    if (img->format != IMAGE_FORMAT_RGB24_DEVICE)
        return image_pad_rgb24_device(img, left, top, right, bottom, RGB);
    assert(0); // todo: fixeme; not implemented
}