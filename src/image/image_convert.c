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
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <mutex>

#define CHECK_NPPcall(call) \
    do { \
        NppStatus _status = (call); \
        if (_status != NPP_SUCCESS) { \
            log_fatal("NPP error code %d", _status); \
            exit(1); \
        } \
    } while (0)

static image_t *image_convert_nv12_to_yuv420_npp(image_t *src, image_format_t format)
{
    image_t *dst=create_image(src->width, src->height, IMAGE_FORMAT_YUV420_DEVICE);
    NppiSize roi={src->width, src->height};
    Npp8u *pSrc[2]={src->y, src->u};
    Npp8u *pDst[3]={dst->y, dst->u, dst->v};
    int aDstStep[3]={dst->stride_y, dst->stride_uv, dst->stride_uv};
    image_add_dependency(dst, src); // don't run this until 'src' is ready
    NppStreamContext nppStreamCtx=get_nppStreamCtx();
    nppStreamCtx.hStream=dst->stream;
    CHECK_NPPcall(nppiNV12ToYUV420_8u_P2P3R_Ctx(pSrc, src->stride_y, pDst, aDstStep, roi, nppStreamCtx));
    return dst;
}

static image_t *image_convert_yuv420_to_nv12_device(image_t *src, image_format_t format)
{
    image_t *dst = create_image(src->width, src->height, IMAGE_FORMAT_NV12_DEVICE);
    image_add_dependency(dst, src);

    int width = src->width;
    int height = src->height;
    int uv_width = width / 2;
    int uv_height = height / 2;

    // Copy Y-plane
    CHECK_CUDART_CALL(cudaMemcpy2DAsync(
        dst->y, dst->stride_y,
        src->y, src->stride_y,
        width, height,
        cudaMemcpyDeviceToDevice,
        dst->stream
    ));

    // Interleave U (Cb) and V (Cr) into NV12 format
    cuda_interleave_uv(
        src->u, src->v,                      // u = Cb, v = Cr
        src->stride_uv,
        dst->u,                              // NV12 interleaved CbCr output
        dst->stride_uv,
        uv_width,
        uv_height,
        dst->stream
    );

    return dst;
}

static inline uint8_t clip_float_to_u8(float val) {
    return (uint8_t)(fminf(fmaxf(val, 0.0f), 255.0f));
}

// BT.709 limited range to RGB conversion
static void yuv2rgb_bt709(uint8_t y, uint8_t u, uint8_t v, uint8_t* r, uint8_t* g, uint8_t* b)
{
    // Limited range assumption
    float yf = (float)y;
    float uf = (float)u - 128.0f;
    float vf = (float)v - 128.0f;

    float c = yf - 16.0f;
    float r_f = 1.164f * c + 1.793f * vf;
    float g_f = 1.164f * c - 0.213f * uf - 0.533f * vf;
    float b_f = 1.164f * c + 2.112f * uf;

    *r = clip_float_to_u8(r_f);
    *g = clip_float_to_u8(g_f);
    *b = clip_float_to_u8(b_f);
}

static image_t* image_convert_yuv420_to_rgb24_host(image_t* src, image_format_t format)
{
    assert(src->format == IMAGE_FORMAT_YUV420_HOST);
    image_t* dst = create_image(src->width, src->height, IMAGE_FORMAT_RGB24_HOST);
    if (!dst) return NULL;

    image_add_dependency(dst, src);

    for (int y = 0; y < src->height; y++) {
        for (int x = 0; x < src->width; x++) {
            uint8_t cy = src->y[y * src->stride_y + x];
            uint8_t cu = src->u[(y >> 1) * src->stride_uv + (x >> 1)];
            uint8_t cv = src->v[(y >> 1) * src->stride_uv + (x >> 1)];

            uint8_t* d = dst->rgb + y * dst->stride_rgb + 3 * x;
            yuv2rgb_bt709(cy, cu, cv, d, d + 1, d + 2);
        }
    }

    return dst;
}

static image_t* image_convert_yuv420_to_rgb24_device(image_t* src, image_format_t format)
{
    assert(src->format == IMAGE_FORMAT_YUV420_DEVICE);
    image_t* dst = create_image(src->width, src->height, IMAGE_FORMAT_RGB24_DEVICE);
    if (!dst) return NULL;

    image_add_dependency(dst, src);

    cuda_convertYUVtoRGB24(src->y, src->u, src->v,
        src->stride_y, src->stride_uv,
        dst->rgb, dst->stride_rgb, src->width, src->height, dst->stream);

    return dst;
}

static image_t *image_convert_rgb24_to_yuv_device(image_t *src, image_format_t format)
{
    image_t *dest=create_image(src->width, src->height, IMAGE_FORMAT_YUV420_DEVICE);
    if (!dest) return 0;

    image_add_dependency(dest, src);

    cuda_convertRGB24toYUV420(src->rgb, src->stride_rgb,
        dest->y, dest->u, dest->v, dest->stride_y, dest->stride_uv,
        dest->width, dest->height, dest->stream);

    return dest;
}

static void copyasync2d(void *src, int src_stride, bool src_device, void *dest, int dest_stride, bool dest_device, int width, int height, CUstream stream)
{
    cudaMemcpyKind kind;
    if (src_device && dest_device)
        kind = cudaMemcpyDeviceToDevice;
    else if (!src_device && dest_device)
        kind = cudaMemcpyHostToDevice;
    else if (src_device && !dest_device)
        kind = cudaMemcpyDeviceToHost;
    else
        kind = cudaMemcpyHostToHost;
    CHECK_CUDART_CALL(cudaMemcpy2DAsync(dest, dest_stride,src, src_stride,width, height,kind, stream));
}

static image_t *image_convert_yuv420_device_host(image_t *img, image_format_t format)
{
    bool src_device=image_format_is_device(img->format);
    bool dest_device=image_format_is_device(format);
    image_t *ret=create_image(img->width, img->height, format);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    copyasync2d(img->y,img->stride_y,src_device,ret->y,ret->stride_y,dest_device,img->width,img->height,ret->stream);
    // checks below make this function also work for MONO surfaces, which is handy
    if (img->u!=0) copyasync2d(img->u,img->stride_uv,src_device,ret->u,ret->stride_uv,dest_device,img->width/2,img->height/2,ret->stream);
    if (img->v!=0) copyasync2d(img->v,img->stride_uv,src_device,ret->v,ret->stride_uv,dest_device,img->width/2,img->height/2,ret->stream);
    return ret;
}

static image_t *image_convert_rgb_planar_fp_device_host(image_t *img, image_format_t format)
{
    bool src_device=image_format_is_device(img->format);
    bool dest_device=image_format_is_device(format);
    int bpf=((format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE)
             ||(format==IMAGE_FORMAT_RGB_PLANAR_FP16_HOST)) ? 2 : 4;

    image_t *ret=create_image(img->width, img->height, dest_device ? IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE : IMAGE_FORMAT_RGB_PLANAR_FP32_HOST);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    CHECK_CUDART_CALL(cudaMemcpyAsync(ret->rgb, img->rgb, img->width*img->height*bpf*3, dest_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, ret->stream));
    return ret;
}

static image_t *image_convert_rgb24_device_host(image_t *img, image_format_t format)
{
    bool src_device=image_format_is_device(img->format);
    bool dest_device=image_format_is_device(format);
    image_t *ret=create_image(img->width, img->height, dest_device ? IMAGE_FORMAT_RGB24_DEVICE : IMAGE_FORMAT_RGB24_HOST);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    copyasync2d(img->rgb,img->stride_rgb,src_device,ret->rgb,ret->stride_rgb,dest_device,img->width*3,img->height,ret->stream);
    return ret;
}

static image_t *image_convert_yuv420_device_planar_rgb_fp16(image_t *img, image_format_t format)
{
    image_t *ret=create_image(img->width, img->height, IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    cuda_convertYUVtoRGB_fp16(img->y, img->u, img->v, img->stride_y, img->stride_uv, ret->rgb, img->width, img->height, ret->stream);
    return ret;
}

static image_t *image_convert_yuv420_device_planar_rgb_fp32(image_t *img, image_format_t format)
{
    image_t *ret=create_image(img->width, img->height, IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    cuda_convertYUVtoRGB_fp32(img->y, img->u, img->v, img->stride_y, img->stride_uv, ret->rgb, img->width, img->height, ret->stream);
    return ret;
}

static image_t *image_planar_fp16_fp32_device(image_t *img, image_format_t format)
{
    image_t *ret=create_image(img->width, img->height, IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    cuda_half_to_float(img->rgb, ret->rgb, img->width*img->height);
    return ret;
}

static image_t *image_convert_planar_fp16_rgb24_device(image_t *img, image_format_t format)
{
    image_t *ret=create_image(img->width, img->height, IMAGE_FORMAT_RGB24_DEVICE);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    cuda_convert_fp16_planar_to_RGB24(img->rgb, ret->rgb, ret->stride_rgb, img->width, img->height, ret->stream);
    return ret;
}

static image_t *image_convert_yuv420_mono(image_t *src, image_format_t format)
{
    // what we do here is not actually any work - we just create a new shell surface
    // that points to the Y data of the original surface, and hold on to a reference
    // for that surface.

    if (true)
    {
        image_t *mono=create_image_no_surface_memory(src->width, src->height, IMAGE_FORMAT_MONO_DEVICE);
        mono->referenced_surface=image_reference(src);
        mono->y=src->y;
        mono->stride_y=src->stride_y;
        image_add_dependency(mono, src);
        return mono;
    }
    else
    {
        image_t *dst = create_image(src->width, src->height, IMAGE_FORMAT_MONO_DEVICE);
        image_add_dependency(dst, src);
        cudaStream_t stream = dst->stream;
        // Async copy Y plane
        CHECK_CUDART_CALL(cudaMemcpy2DAsync(dst->y, dst->stride_y,src->y, src->stride_y,
            src->width, src->height,cudaMemcpyDeviceToDevice,dst->stream));
        return dst;
    }
}

static image_t *image_convert_rgb24_planar_fp_device(image_t *src, image_format_t format)
{
    image_t *dst = create_image(src->width, src->height, format);
    assert(format==IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE);
    image_add_dependency(dst, src);
    cuda_convert_rgb24_to_planar_fp32(src->rgb, (float*)dst->rgb,
                        src->width, src->height, src->stride_rgb, dst->stream);
    return dst;
}

static image_t *image_convert_mono_yuv420(image_t *src, image_format_t format)
{
    image_t *dst = create_image(src->width, src->height, IMAGE_FORMAT_YUV420_DEVICE);
    image_add_dependency(dst, src);
    cudaStream_t stream = dst->stream;
    // Async copy Y plane
    CHECK_CUDART_CALL(cudaMemcpy2DAsync(dst->y, dst->stride_y,src->y, src->stride_y,
        src->width, src->height,cudaMemcpyDeviceToDevice,dst->stream));
    // Calculate size of U/V planes (half width and height)
    int uv_width = src->width / 2;
    int uv_height = src->height / 2;
    // Async memset U and V planes to 0x80 (neutral chroma)
    CHECK_CUDART_CALL(cudaMemset2DAsync(dst->u, dst->stride_uv, 0x80, uv_width, uv_height, dst->stream));
    CHECK_CUDART_CALL(cudaMemset2DAsync(dst->v, dst->stride_uv, 0x80, uv_width, uv_height, dst->stream));
    return dst;
}

typedef struct image_conversion_method
{
    image_format_t src;
    image_format_t dest;
    image_t *(*convert_direct)(image_t *img, image_format_t format);
    image_format_t convert_intermediate;
    int cost;
} image_conversion_method_t;

static image_conversion_method_t direct_methods[] = {
    {IMAGE_FORMAT_NV12_DEVICE, IMAGE_FORMAT_YUV420_DEVICE, image_convert_nv12_to_yuv420_npp, IMAGE_FORMAT_NONE, 100},
    {IMAGE_FORMAT_YUV420_DEVICE, IMAGE_FORMAT_NV12_DEVICE, image_convert_yuv420_to_nv12_device, IMAGE_FORMAT_NONE, 100},
    {IMAGE_FORMAT_YUV420_HOST, IMAGE_FORMAT_YUV420_DEVICE, image_convert_yuv420_device_host, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_YUV420_DEVICE, IMAGE_FORMAT_YUV420_HOST, image_convert_yuv420_device_host, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_RGB24_HOST, IMAGE_FORMAT_RGB24_DEVICE, image_convert_rgb24_device_host, IMAGE_FORMAT_NONE, 30},
    {IMAGE_FORMAT_RGB24_DEVICE, IMAGE_FORMAT_RGB24_HOST, image_convert_rgb24_device_host, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_YUV420_HOST, IMAGE_FORMAT_RGB24_HOST, image_convert_yuv420_to_rgb24_host, IMAGE_FORMAT_NONE, 200},
    {IMAGE_FORMAT_YUV420_DEVICE, IMAGE_FORMAT_RGB24_DEVICE, image_convert_yuv420_to_rgb24_device, IMAGE_FORMAT_NONE, 100},
    {IMAGE_FORMAT_RGB24_DEVICE, IMAGE_FORMAT_YUV420_DEVICE, image_convert_rgb24_to_yuv_device, IMAGE_FORMAT_NONE, 100},
    {IMAGE_FORMAT_YUV420_DEVICE, IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE, image_convert_yuv420_device_planar_rgb_fp16, IMAGE_FORMAT_NONE, 120},
    {IMAGE_FORMAT_YUV420_DEVICE, IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE, image_convert_yuv420_device_planar_rgb_fp32, IMAGE_FORMAT_NONE, 120},
    {IMAGE_FORMAT_RGB_PLANAR_FP32_HOST, IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE, image_convert_rgb_planar_fp_device_host, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_RGB24_DEVICE, IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE, image_convert_rgb24_planar_fp_device, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE, IMAGE_FORMAT_RGB_PLANAR_FP32_HOST, image_convert_rgb_planar_fp_device_host, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_RGB_PLANAR_FP16_HOST, IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE, image_convert_rgb_planar_fp_device_host, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE, IMAGE_FORMAT_RGB_PLANAR_FP16_HOST, image_convert_rgb_planar_fp_device_host, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE, IMAGE_FORMAT_RGB24_DEVICE, image_convert_planar_fp16_rgb24_device, IMAGE_FORMAT_NONE, 110},
    {IMAGE_FORMAT_YUV420_DEVICE, IMAGE_FORMAT_MONO_DEVICE, image_convert_yuv420_mono, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_MONO_DEVICE, IMAGE_FORMAT_YUV420_DEVICE, image_convert_mono_yuv420, IMAGE_FORMAT_NONE, 100},
    {IMAGE_FORMAT_MONO_HOST, IMAGE_FORMAT_MONO_DEVICE, image_convert_yuv420_device_host, IMAGE_FORMAT_NONE, 50},
    {IMAGE_FORMAT_MONO_DEVICE, IMAGE_FORMAT_MONO_HOST, image_convert_yuv420_device_host, IMAGE_FORMAT_NONE, 50}
};

static image_conversion_method_t* conversion_table[NUM_IMAGE_FORMATS][NUM_IMAGE_FORMATS] = {0};

void image_conversion_init()
{
    int n_methods = sizeof(direct_methods) / sizeof(direct_methods[0]);
    log_debug("Image conversion init");
    // Phase 1: Insert direct methods
    for (int i = 0; i < n_methods; i++) {
        image_conversion_method_t *m = &direct_methods[i];
        conversion_table[m->src][m->dest] = m;
    }
    // Phase 2: Add conversions via intermediates
    for (int iter=0;iter<3;iter++) {
        for (int src = 0; src < NUM_IMAGE_FORMATS; ++src) {
            for (int dest = 0; dest < NUM_IMAGE_FORMATS; ++dest) {
                if (src == dest || conversion_table[src][dest])
                    continue;  // already handled

                int best_cost = 1e9;
                image_format_t best_intermediate = IMAGE_FORMAT_NONE;

                for (int mid = 0; mid < NUM_IMAGE_FORMATS; ++mid) {
                    if (mid == src || mid == dest)
                        continue;

                    image_conversion_method_t *m1 = conversion_table[src][mid];
                    image_conversion_method_t *m2 = conversion_table[mid][dest];
                    if (m1 && m2) {
                        int total_cost = m1->cost + m2->cost;
                        if (total_cost < best_cost) {best_cost = total_cost;best_intermediate = (image_format_t)mid; }
                    }
                }

                if (best_intermediate != IMAGE_FORMAT_NONE)
                {
                    image_conversion_method_t *new_entry = (image_conversion_method_t *)malloc(sizeof(image_conversion_method_t));
                    memset(new_entry, 0, sizeof(image_conversion_method_t));
                    new_entry->src = (image_format_t)src;
                    new_entry->dest = (image_format_t)dest;
                    new_entry->convert_intermediate = best_intermediate;
                    new_entry->cost = best_cost;
                    conversion_table[src][dest] = new_entry;
                }
            }
        }
    }
}

image_t *image_convert(image_t *img, image_format_t format)
{
    //log_debug("convert %s->%s",image_format_name(img->format),image_format_name(format));

    if (format==img->format) return image_reference(img);

    image_conversion_method_t *c=conversion_table[img->format][format];

    if (c->convert_direct!=0)
    {
        //log_debug("convert direct");
        image_t *ret=c->convert_direct(img, format);
        //log_debug("ok");
        return ret;
    }
    else if (c->convert_intermediate!=IMAGE_FORMAT_NONE)
    {
        //log_debug("convert intermediate");
        image_t *intermediate=image_convert(img, c->convert_intermediate);
        image_t *ret=image_convert(intermediate, format);
        destroy_image(intermediate);
        //log_debug("OK!");
        return ret;
    }

    printf("Cannot convert %s->%s\n",image_format_name(img->format),image_format_name(format));
    return 0; // unsupported conversion
}
