#include <npp.h>
#include <cuda.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "image.h"
#include "libyuv.h"
#include "cuda_stuff.h"
#include "cuda_kernels.h"

static image_t *image_scale_yuv420_host(image_t *src, int width, int height)
{
    image_t *dst=create_image(width, height, IMAGE_FORMAT_YUV420_HOST);
    if (!dst) return 0;

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
    image_t *dst=create_image(width, height, IMAGE_FORMAT_YUV420_DEVICE);
    if (!dst) return 0;

    NppiSize srcSize = {src->width, src->height};
    NppiRect srcROI = {0, 0, src->width, src->height};
    NppiRect dstROI = {0, 0, dst->width, dst->height};

    image_add_dependency(dst, src); // don't run this until 'src' is ready
    NppStreamContext nppStreamCtx=get_nppStreamCtx();
    nppStreamCtx.hStream=dst->stream;

    // Create scaling context for Y plane
    NppiInterpolationMode interpolationMode = NPPI_INTER_LANCZOS;
    NppiSize dstSize = {dst->width, dst->height};

    // Y plane scaling
    NppStatus result = nppiResize_8u_C1R_Ctx(src->y, src->stride_y, srcSize, srcROI,
                                         dst->y, dst->stride_y, dstSize, dstROI,
                                         interpolationMode, nppStreamCtx);
    assert(result== NPP_SUCCESS);

    // Scale U and V planes (quarter resolution of Y in YUV420)
    NppiSize srcSizeUV = {src->width / 2, src->height / 2};
    NppiRect srcROIUV = {0, 0, src->width / 2, src->height / 2};
    NppiSize dstSizeUV = {dst->width / 2, dst->height / 2};
    NppiRect dstROIUV = {0, 0, dst->width / 2, dst->height / 2};

    // U plane scaling
    result = nppiResize_8u_C1R_Ctx(src->u, src->stride_uv, srcSizeUV, srcROIUV,
                               dst->u, dst->stride_uv, dstSizeUV, dstROIUV,
                               interpolationMode, nppStreamCtx);
    assert(result== NPP_SUCCESS);

    // V plane scaling
    result = nppiResize_8u_C1R_Ctx(src->v, src->stride_uv, srcSizeUV, srcROIUV,
                               dst->v, dst->stride_uv, dstSizeUV, dstROIUV,
                               interpolationMode, nppStreamCtx);
    assert(result== NPP_SUCCESS); 
    return dst;
}

static image_t *image_scale_by_intermediate(image_t *img, int width, int height, image_format_t inter)
{
    assert(img->format!=inter);
    image_t *image_tmp=image_convert(img, inter);
    image_t *scaled=image_scale(image_tmp, width, height);
    destroy_image(image_tmp);
    return scaled;
}

image_t *image_scale(image_t *img, int width, int height)
{
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
        default:
        ;
    }
    return 0;
}

static image_t *image_convert_nv12_to_yuv420_npp(image_t *src)
{
    image_t *dst=create_image(src->width, src->height, IMAGE_FORMAT_YUV420_DEVICE);
    NppiSize roi={src->width, src->height};
    Npp8u *pSrc[2]={src->y, src->u};
    Npp8u *pDst[3]={dst->y, dst->u, dst->v};
    int aDstStep[3]={dst->stride_y, dst->stride_uv, dst->stride_uv};
    image_add_dependency(dst, src); // don't run this until 'src' is ready
    NppStreamContext nppStreamCtx=get_nppStreamCtx();
    nppStreamCtx.hStream=dst->stream;
    NppStatus result=nppiNV12ToYUV420_8u_P2P3R_Ctx(pSrc, src->stride_y, pDst, aDstStep, roi, nppStreamCtx);
    return dst;
}

static image_t *image_convert_yuv420_to_nv12_npp(image_t *src)
{
    image_t *dst = create_image(src->width, src->height, IMAGE_FORMAT_NV12_DEVICE);
    image_add_dependency(dst, src);

    NppiSize roi = { src->width, src->height };
    NppStreamContext nppStreamCtx = get_nppStreamCtx();
    nppStreamCtx.hStream = dst->stream;

    const Npp8u *pSrc[3] = { src->y, src->u, src->v }; // assuming src->u = Cr, src->v = Cb
    int aSrcStep[3] = { src->stride_y, src->stride_uv, src->stride_uv };

    Npp8u *pDstY = dst->y;
    Npp8u *pDstCbCr = dst->u; // interleaved CbCr buffer for NV12
    int dstStepY = dst->stride_y;
    int dstStepCbCr = dst->stride_uv;

    NppStatus result = nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(
        pSrc, aSrcStep,
        pDstY, dstStepY,
        pDstCbCr, dstStepCbCr,
        roi, nppStreamCtx);

    return dst;
}

static int clip(int x)
{
    if (x<0) return 0;
    if (x>255) return 255;
    return x;
}

static void yuv2rgb(unsigned char y, unsigned char u, unsigned char v, unsigned char *r, unsigned char *g, unsigned char *b) 
{
    int c = y - 16;
    int d = u - 128;
    int e = v - 128;
    *r = (unsigned char) (clip(( 298 * c + 409 * e + 128) >> 8));
    *g = (unsigned char) (clip(( 298 * c - 100 * d - 208 * e + 128) >> 8));
    *b = (unsigned char) (clip(( 298 * c + 516 * d + 128) >> 8));
}

static image_t *image_convert_yuv420_to_rgb24_host(image_t *src)
{
    image_t *dst=create_image(src->width, src->height, IMAGE_FORMAT_RGB24_HOST);
    if (!dst) return 0;
    image_add_dependency(dst, src);
    for(int y=0;y<src->height;y++)
    {
        for(int x=0;x<src->width;x++)
        {
            uint8_t cy=src->y[x+y*src->stride_y];
            uint8_t cu=src->u[(x>>1)+(y>>1)*src->stride_uv];
            uint8_t cv=src->v[(x>>1)+(y>>1)*src->stride_uv];
            uint8_t *d=dst->rgb+y*dst->stride_rgb+3*x;
            yuv2rgb(cy,cu,cv,d+0,d+1,d+2);
        }
    }
    return dst;
}

static image_t *image_convert_rgb24_to_yuv_device(image_t *src)
{
    image_t *dest=create_image(src->width, src->height, IMAGE_FORMAT_YUV420_DEVICE);
    if (!dest) return 0;
    image_add_dependency(dest, src);
    Npp8u *pDst[3];
    
    NppiSize oSizeROI = {src->width, src->height};
    int nDstYStep = dest->stride_y;    // Y channel step
    int nDstUVStep = dest->stride_uv; // U and V channel step
    int aDstStep[3]={nDstYStep,nDstUVStep,nDstUVStep};

    pDst[0] = dest->y;
    pDst[1] = dest->u;
    pDst[2] = dest->v;

    // Conversion using NPP function
    image_add_dependency(dest, src); // don't run this until 'src' is ready
    NppStreamContext nppStreamCtx=get_nppStreamCtx();
    nppStreamCtx.hStream=dest->stream;
    nppiRGBToYUV420_8u_C3P3R_Ctx(src->rgb, src->stride_rgb, pDst, aDstStep, oSizeROI, nppStreamCtx);
    return dest;
}

static void copyasync2d(void *src, int src_stride, bool src_device, void *dest, int dest_stride, bool dest_device, int width, int height, CUstream stream)
{
    CUDA_MEMCPY2D copyP;
    memset(&copyP, 0, sizeof(copyP));
    //printf("COPY2D %dx%d ss %d ds %d\n",width,height,src_stride,dest_stride);
    if (src_device)
    {
        copyP.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copyP.srcDevice = (CUdeviceptr)src;
    } 
    else
    {
        copyP.srcMemoryType = CU_MEMORYTYPE_HOST;
        copyP.srcHost = src;
    }
    if (dest_device)
    {
        copyP.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copyP.dstDevice = (CUdeviceptr)dest;
    } 
    else
    {
        copyP.dstMemoryType = CU_MEMORYTYPE_HOST;
        copyP.dstHost = dest;
    }
    copyP.srcPitch = src_stride;
    copyP.dstPitch = dest_stride;
    copyP.WidthInBytes = width;
    copyP.Height = height;
    assert(CUDA_SUCCESS==cuMemcpy2DAsync(&copyP, stream));
}

static image_t *image_convert_yuv420_device_host(image_t *img, bool src_device, bool dest_device)
{
    image_t *ret=create_image(img->width, img->height, dest_device ? IMAGE_FORMAT_YUV420_DEVICE : IMAGE_FORMAT_YUV420_HOST);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    copyasync2d(img->y,img->stride_y,src_device,ret->y,ret->stride_y,dest_device,img->width,img->height,img->stream);
    copyasync2d(img->u,img->stride_uv,src_device,ret->u,ret->stride_uv,dest_device,img->width/2,img->height/2,img->stream);
    copyasync2d(img->v,img->stride_uv,src_device,ret->v,ret->stride_uv,dest_device,img->width/2,img->height/2,img->stream);
    return ret;
}

static image_t *image_convert_rgb_planar_fp_device_host(image_t *img, bool src_device, bool dest_device, int bpf)
{
    image_t *ret=create_image(img->width, img->height, dest_device ? IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE : IMAGE_FORMAT_RGB_PLANAR_FP32_HOST);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    if (dest_device && (!src_device))
    {
        cuMemcpyHtoDAsync((CUdeviceptr)ret->rgb, (const void *)img->rgb, img->width*img->height*bpf*3, ret->stream); 
    }
    else if ((!dest_device) && (src_device))
    {
        cuMemcpyDtoHAsync((void*)ret->rgb, (CUdeviceptr)img->rgb, img->width*img->height*bpf*3, img->stream); 
    }
    return ret;
}

static image_t *image_convert_rgb24_device_host(image_t *img, bool src_device, bool dest_device)
{
    image_t *ret=create_image(img->width, img->height, dest_device ? IMAGE_FORMAT_RGB24_DEVICE : IMAGE_FORMAT_RGB24_HOST);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    copyasync2d(img->rgb,img->stride_rgb,src_device,ret->rgb,ret->stride_rgb,dest_device,img->width*3,img->height,img->stream);
    if (dest_device==false)
    {
        for(int i=0;i<100;i++) ret->rgb[3*i+i*ret->stride_rgb]=0xd0;
    }
    return ret;
}

static image_t *image_convert_yuv420_device_planar_rgb_fp16(image_t *img)
{
    image_t *ret=create_image(img->width, img->height, IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    cuda_convertYUVtoRGB_fp16(img->y, img->u, img->v, img->stride_y, img->stride_uv, ret->rgb, img->width, img->height, img->stream);
    return ret;
}

static image_t *image_convert_yuv420_device_planar_rgb_fp32(image_t *img)
{
    image_t *ret=create_image(img->width, img->height, IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    cuda_convertYUVtoRGB_fp32(img->y, img->u, img->v, img->stride_y, img->stride_uv, ret->rgb, img->width, img->height, img->stream);
    return ret;
}

static image_t *image_planar_fp16_fp32_device(image_t *img)
{
    image_t *ret=create_image(img->width, img->height, IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    cuda_half_to_float(img->rgb, ret->rgb, img->width*img->height);
    return ret;
}

static image_t *image_convert_planar_fp16_rgb24_device(image_t *img)
{
    image_t *ret=create_image(img->width, img->height, IMAGE_FORMAT_RGB24_DEVICE);
    if (!ret) return 0;
    image_add_dependency(ret, img);
    cuda_convert_fp16_planar_to_RGB24(img->rgb, ret->rgb, ret->stride_rgb, img->width, img->height, img->stream);
    return ret;
}

static image_t *image_convert_via_intermediate(image_t *img, image_format_t intermediate, image_format_t format)
{
    image_t *temp=image_convert(img, intermediate);
    image_t *ret=image_convert(temp, format);
    destroy_image(temp);
    return ret;
}

image_t *image_convert(image_t *img, image_format_t format)
{
    //printf("convert %s->%s\n",image_format_name(img->format),image_format_name(format));

    if (format==img->format) return image_reference(img);

    if ((format==IMAGE_FORMAT_YUV420_DEVICE)&&(img->format==IMAGE_FORMAT_NV12_DEVICE))
        return image_convert_nv12_to_yuv420_npp(img);

    if ((format==IMAGE_FORMAT_NV12_DEVICE)&&(img->format==IMAGE_FORMAT_YUV420_DEVICE))
        return image_convert_nv12_to_yuv420_npp(img);

    if ((format==IMAGE_FORMAT_YUV420_HOST)&&(img->format==IMAGE_FORMAT_YUV420_DEVICE))
        return image_convert_yuv420_device_host(img, true, false);

    if ((format==IMAGE_FORMAT_YUV420_DEVICE)&&(img->format==IMAGE_FORMAT_YUV420_HOST))
        return image_convert_yuv420_device_host(img, false, true);

    if ((format==IMAGE_FORMAT_RGB24_HOST)&&(img->format==IMAGE_FORMAT_RGB24_DEVICE))
        return image_convert_rgb24_device_host(img, true, false);

    if ((format==IMAGE_FORMAT_RGB24_DEVICE)&&(img->format==IMAGE_FORMAT_RGB24_HOST))
        return image_convert_rgb24_device_host(img, false, true);

    if ((format==IMAGE_FORMAT_RGB24_HOST)&&(img->format==IMAGE_FORMAT_YUV420_HOST))
        return image_convert_yuv420_to_rgb24_host(img);

    if ((format==IMAGE_FORMAT_RGB24_HOST)&&(img->format==IMAGE_FORMAT_YUV420_DEVICE))
        return image_convert_via_intermediate(img, IMAGE_FORMAT_YUV420_HOST, format);

    if ((format==IMAGE_FORMAT_YUV420_DEVICE)&&(img->format==IMAGE_FORMAT_RGB24_DEVICE))
        return image_convert_rgb24_to_yuv_device(img);

    if ((format==IMAGE_FORMAT_YUV420_DEVICE)&&(img->format==IMAGE_FORMAT_RGB24_HOST))
        return image_convert_via_intermediate(img, IMAGE_FORMAT_RGB24_DEVICE, format);

    if ((format==IMAGE_FORMAT_YUV420_DEVICE)&&(img->format==IMAGE_FORMAT_RGB24_HOST))
        return image_convert_via_intermediate(img, IMAGE_FORMAT_YUV420_HOST, format);

    if ((format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE)&&(img->format==IMAGE_FORMAT_YUV420_DEVICE))
        return image_convert_yuv420_device_planar_rgb_fp16(img);

    if ((format==IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE)&&(img->format==IMAGE_FORMAT_YUV420_DEVICE))
        return image_convert_yuv420_device_planar_rgb_fp32(img);

    if ((format==IMAGE_FORMAT_RGB_PLANAR_FP32_HOST)&&(img->format==IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE))
        return image_convert_rgb_planar_fp_device_host(img, true, false, 4);

    if ((format==IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE)&&(img->format==IMAGE_FORMAT_RGB_PLANAR_FP32_HOST))
        return image_convert_rgb_planar_fp_device_host(img, false, true, 4);

    if ((format==IMAGE_FORMAT_RGB_PLANAR_FP16_HOST)&&(img->format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE))
        return image_convert_rgb_planar_fp_device_host(img, true, false, 2);

    if ((format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE)&&(img->format==IMAGE_FORMAT_RGB_PLANAR_FP16_HOST))
        return image_convert_rgb_planar_fp_device_host(img, false, true, 2);

    if ((format==IMAGE_FORMAT_RGB24_DEVICE)&&(img->format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE))
        return image_convert_planar_fp16_rgb24_device(img);
    
    if ((format==IMAGE_FORMAT_RGB24_HOST)&&(img->format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE))
        return image_convert_via_intermediate(img, IMAGE_FORMAT_RGB24_DEVICE, format);

    printf("Cannot convert %s->%s\n",image_format_name(img->format),image_format_name(format));
    return 0; // unsupported conversion
}