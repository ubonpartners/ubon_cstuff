#include <npp.h>
#include <cuda.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include "image.h"
#include "libyuv.h"
#include "cuda_stuff.h"
#include "cuda_kernels.h"
#include "misc.h"
#include "display.h"
#include "solvers.h"
#include <mutex>
#include <math.h>

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
    CHECK_CUDART_CALL(cudaMemcpy2DAsync(
    out->y + dy * out->stride_y + dx,         // dst pointer
    out->stride_y,                            // dst pitch
    src2->y + sy * src2->stride_y + sx,       // src pointer
    src2->stride_y,                           // src pitch
    w,                                        // width in bytes
    h,                                        // height
    cudaMemcpyDeviceToDevice,                 // direction
    out->stream));

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

        // Copy U block
        CHECK_CUDART_CALL(cudaMemcpy2DAsync(
            out->u + dy_uv * out->stride_uv + dx_uv,      // dst pointer
            out->stride_uv,                               // dst pitch
            src2->u + sy_uv * src2->stride_uv + sx_uv,    // src pointer
            src2->stride_uv,                              // src pitch
            uv_w,                                         // width in bytes
            uv_h,                                         // height
            cudaMemcpyDeviceToDevice,                     // direction
            out->stream));                                // stream

        // Copy V block
        CHECK_CUDART_CALL(cudaMemcpy2DAsync(
            out->v + dy_uv * out->stride_uv + dx_uv,      // dst pointer
            out->stride_uv,                               // dst pitch
            src2->v + sy_uv * src2->stride_uv + sx_uv,    // src pointer
            src2->stride_uv,                              // src pitch
            uv_w,                                         // width in bytes
            uv_h,                                         // height
            cudaMemcpyDeviceToDevice,                     // direction
            out->stream));                                // stream
    }

    return out;
}

image_t *image_crop(image_t *img, int x, int y, int w, int h)
{
    if (!img) return 0;
    if (   img->format != IMAGE_FORMAT_YUV420_DEVICE
        && img->format != IMAGE_FORMAT_MONO_DEVICE
        && img->format != IMAGE_FORMAT_RGB24_DEVICE)
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
    image_add_dependency(cropped, img);

    if ( (cropped->format==IMAGE_FORMAT_RGB24_DEVICE)
       ||(cropped->format==IMAGE_FORMAT_RGB24_HOST))
    {
        cropped->rgb=img->rgb+x*3+y*img->stride_y;
        cropped->stride_rgb=img->stride_rgb;
    }
    else // MONO_ or YUV420_
    {
        cropped->y=img->y+x+y*img->stride_y;
        cropped->stride_y=img->stride_y;
        if (  (cropped->format==IMAGE_FORMAT_YUV420_DEVICE)
            ||(cropped->format==IMAGE_FORMAT_YUV420_HOST))
        {
            cropped->stride_uv=img->stride_uv;
            cropped->u=img->u+(x>>1)+(y>>1)*cropped->stride_uv;
            cropped->v=img->v+(x>>1)+(y>>1)*cropped->stride_uv;
        }
    }

    return cropped;
}


image_t *image_crop_roi(image_t *img, roi_t in_roi, roi_t *out_roi)
{
    if (in_roi.box[0]<=0 && in_roi.box[1]<=0 && in_roi.box[2]>=1 && in_roi.box[3]>=1)
    {
        *out_roi=in_roi;
        return image_reference(img);
    }
    int x0=(int)(in_roi.box[0]*img->width);
    int y0=(int)(in_roi.box[1]*img->height);
    int x1=(int)(ceilf(in_roi.box[2]*img->width));
    int y1=(int)(ceilf(in_roi.box[3]*img->height));

    // round to even (making crop bigger)
    // todo: only do this for chroma downsampled formats
    x0=std::max(0, x0&(~1));
    y0=std::max(0, y0&(~1));
    x1=std::min(img->width, (x1+1)&(~1));
    y1=std::min(img->height, (y1+1)&(~1));

    int w=std::max(0, x1-x0);
    int h=std::max(0, y1-y0);
    w=std::min(w, img->width-x0);
    h=std::min(h, img->height-y0);

    image_t *ret=image_crop(img, x0, y0, w, h);

    out_roi->box[0]=((float)x0)/img->width;
    out_roi->box[1]=((float)y0)/img->height;
    out_roi->box[2]=((float)(x0+w))/img->width;
    out_roi->box[3]=((float)(y0+h))/img->height;
    return ret;
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


void image_apply_padding(image_t *img, int pad_l, int pad_t, int pad_r, int pad_b)
{
    assert(img->format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE || img->format==IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE);
    int elt_size=(img->format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE) ? 2 : 4;
    bool is_fp16=(img->format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE);
    int dst_width=img->width;
    int dst_height=img->height;
    if (pad_t>0)
    {
        cuda_fp_set(img->rgb+elt_size*(0*dst_width+0),
                    dst_width, pad_t,
                    dst_width, dst_width*dst_height, img->stream, is_fp16);
    }
    if (pad_b>0)
    {
        cuda_fp_set(img->rgb+elt_size*((img->height-pad_b)*dst_width+0),
                    dst_width, pad_b,
                    dst_width, dst_width*dst_height, img->stream, is_fp16);
    }
    if (pad_l>0)
    {
        cuda_fp_set(img->rgb+elt_size*(pad_t*dst_width+0),
                    pad_l, dst_height-pad_t-pad_b,
                    dst_width, dst_width*dst_height, img->stream, is_fp16);
    }
    if (pad_r>0)
    {
        cuda_fp_set(img->rgb+elt_size*(pad_t*dst_width+dst_width-pad_r),
                    pad_r, dst_height-pad_t-pad_b,
                    dst_width, dst_width*dst_height, img->stream, is_fp16);
    }
}

// image_make_tiled: given N images <= WxH make a single tiled image
// with all the sub-images together
// this is commonly used for batch inference

// following what ultralytics do, we put the image in the middle, and
// use grey padding

image_t *image_make_tiled(image_format_t fmt,
                          int dst_width, int dst_height,
                          image_t **images, int num,
                          int *offs_x, int *offs_y)
{
    assert(fmt==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE || fmt==IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE);
    int elt_size=(fmt==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE) ? 2 : 4;
    image_t *inf_image=create_image(dst_width, dst_height*num, fmt); // image big enough to hold "num" planar RGB images
    image_t *inf_subimage=create_image_no_surface_memory(dst_width, dst_height, fmt);
    for(int i=0;i<num;i++)
    {
        image_t *img=images[i];
        inf_subimage->rgb=inf_image->rgb+3*i*elt_size*dst_width*dst_height;
        inf_subimage->stride_rgb=dst_width*dst_height;

        int src_width=img->width;
        int src_height=img->height;

        int pad_l=((dst_width-src_width)>>1);
        int pad_t=((dst_height-src_height)>>1);

        int pad_r=dst_width-src_width-pad_l;
        int pad_b=dst_height-src_height-pad_t;

        offs_x[i]=pad_l;
        offs_y[i]=pad_t;

        assert(src_width<=dst_width && src_height<=dst_height);

        assert(img->format==IMAGE_FORMAT_RGB24_DEVICE || img->format==IMAGE_FORMAT_YUV420_DEVICE);
        image_add_dependency(inf_image, img);

        // pointer to (pad_l, pad_t) in this destination subimage
        void *offset_dest=inf_image->rgb+3*i*elt_size*dst_width*dst_height
                                          +pad_t*dst_width*elt_size
                                          +pad_l*elt_size;

        if (img->format==IMAGE_FORMAT_RGB24_DEVICE)
        {
            cuda_convert_rgb24_to_fp_planar(img->rgb, img->stride_rgb, src_width, src_height,
                                    offset_dest, dst_width, dst_width*dst_height,
                                    inf_image->stream, fmt==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE);

        }
        else
        {
            cuda_convert_yuv420_to_fp_planar(img->y, img->u, img->v, img->stride_y, img->stride_uv,
                                    offset_dest, dst_width, dst_width*dst_height,
                                    src_width, src_height,
                                    inf_image->stream, fmt==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE);
        }

        image_apply_padding(inf_subimage, pad_l, pad_t, pad_r, pad_b);

    }

    /*for(int i=0;i<num;i++)
    {
        inf_subimage->rgb=inf_image->rgb+3*i*elt_size*dst_width*dst_height;
        display_image("subimage", inf_subimage);
        usleep(5*1000*1000);
    }*/

    destroy_image(inf_subimage);
    return inf_image;
}

void image_get_aligned_faces(image_t **images, float *face_points, int n, int w, int h, image_t **ret)
{
    image_t *temp_in[n];
    for(int i=0;i<n;i++) temp_in[i]=image_convert(images[i], IMAGE_FORMAT_YUV420_DEVICE);

    image_t *img_rgb=create_image(w, h*n, IMAGE_FORMAT_RGB24_DEVICE);
    for(int i=0;i<n;i++)
    {
        ret[i]=create_image_no_surface_memory(w, h, IMAGE_FORMAT_RGB24_DEVICE);
        ret[i]=image_reference(img_rgb);
        image_add_dependency(ret[i], img_rgb);
        ret[i]->rgb=img_rgb->rgb+w*h*3*i;
    }

    float* M = (float*)malloc(n * 6 * sizeof(float));
    solve_affine_face_points(images, face_points, n, w, h, M);

    // warp (cast to const)
    cuda_warp_yuv420_to_planar_float(
        (const image_t**)temp_in,
        img_rgb->rgb, n,
        w, h,
        M, true, false,
        img_rgb->stream
    );
    free(M);
    destroy_image(img_rgb);

    for(int i=0;i<n;i++) destroy_image(temp_in[i]);
}

void determine_scale_size(int w, int h, int max_w, int max_h, int *res_w, int *res_h,
                          int percent_stretch_allowed,
                          int round_w, int round_h,
                          bool allow_upscale)
{
    if (allow_upscale)
    {
        int scale_w_num = max_w;
        int scale_w_den = w;
        int scale_h_num = max_h;
        int scale_h_den = h;

        // Compare scale_w = scale_w_num / scale_w_den vs scale_h = scale_h_num / scale_h_den
        if (scale_w_num * scale_h_den < scale_h_num * scale_w_den) {
            // Use scale_w
            *res_w = max_w;
            *res_h = (h * max_w) / w;
        } else {
            // Use scale_h
            *res_w = (w * max_h) / h;
            *res_h = max_h;
        }
        return;
    }

    // starting image w*h we need to determine a size to scale the image to so that it fits in
    // max_w*max_h. We don't want to distort the aspect ratio too much, nor ever upscale
    int rw=w;
    int rh=h;
    if (rw>max_w)
    {
        // scale by max_w/w
        rh=(rh*max_w)/rw;
        rw=(rw*max_w)/rw;
    }
    if (rh>max_h)
    {
        // scale by max_h/rh
        rw=(rw*max_h)/rh;
        rh=(rh*max_h)/rh;
    }
    if (round_w!=0)
    {
        rw+=(round_w>>1);
        rw&=(~(round_w-1));
        if (rw>max_w) rw-=round_w;
    }
    if (round_h!=0)
    {
        rh+=(round_h>>1);
        rh&=(~(round_h-1));
        if (rh>max_h) rh-=round_h;
    }
    if (percent_stretch_allowed!=0)
    {
        // allow 10% or so distortion if it makes it fit better
        int thr_w=(max_w*(100-percent_stretch_allowed))/100;
        int thr_h=(max_h*(100-percent_stretch_allowed))/100;
        if (rw>thr_w && w>=max_w) rw=max_w;
        if (rh>thr_w && h>=max_h) rh=max_h;
    }
    assert(rw<=max_w);
    assert(rh<=max_h);
    *res_w=rw;
    *res_h=rh;
}

image_t *image_scale_convert(image_t *img, image_format_t format, int width, int height)
{
    image_t *tmp=image_convert(img, format);
    image_t *scaled=image_scale(tmp, width, height);
    destroy_image(tmp);
    return scaled;
}