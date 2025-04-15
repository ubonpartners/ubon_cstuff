#include <stdint.h>
#include <malloc.h>
#include <string.h>
#include "image.h"
#include "cuda_stuff.h"
#include <cuda.h>
#include <assert.h>

static void allocate_image_surfaces(image_t *img)
{
    if (!img) return;
    switch(img->format)
    {
        case IMAGE_FORMAT_YUV420_HOST:
        {
            int round=(img->width+31)&(~31);
            CHECK_CUDA_CALL(cuMemAllocHost(&img->host_mem, round*img->height*3/2));
            img->y=(uint8_t *)img->host_mem;
            img->u=img->y+img->width*img->height;
            img->v=img->u+((img->width*img->height)>>2);
            img->stride_y=img->width;
            img->stride_uv=img->width/2;
            break;
        }
        case IMAGE_FORMAT_RGB24_HOST:
        {
            int round=(img->width+31)&(~31);
            CHECK_CUDA_CALL(cuMemAllocHost(&img->host_mem, round*img->height*3));
            img->rgb=(uint8_t *)img->host_mem;
            img->stride_rgb=img->width*3;
            break;
        }
        case IMAGE_FORMAT_RGB24_DEVICE:
        {
            int round=(img->width+31)&(~31);
            CHECK_CUDA_CALL(cuMemAlloc(&img->device_mem, round*img->height*3));
            img->rgb=(uint8_t *)img->device_mem;
            img->stride_rgb=img->width*3;
            break;
        }
        case IMAGE_FORMAT_YUV420_DEVICE:
        {
            int round=(img->width+31)&(~31);
            CHECK_CUDA_CALL(cuMemAlloc(&img->device_mem, round*img->height*3/2));
            img->y=(uint8_t *)img->device_mem;
            img->u=img->y+(round*img->height);
            img->v=img->u+((round*img->height)>>2);
            img->stride_y=round;
            img->stride_uv=round>>1;
            break;
        }
        case IMAGE_FORMAT_NV12_DEVICE:
        {
            int round=(img->width+31)&(~31);
            CHECK_CUDA_CALL(cuMemAlloc(&img->device_mem, round*img->height*3/2));
            img->y=(uint8_t *)img->device_mem;
            img->u=img->y+(round*img->height);
            img->v=img->u+1;
            img->stride_y=round;
            img->stride_uv=round;
            break;
        }
        case IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE:
        {
            CHECK_CUDA_CALL(cuMemAlloc(&img->device_mem, img->width*img->height*3*2));
            img->rgb=(uint8_t *)img->device_mem;
            break;
        }
        case IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE:
        {
            CHECK_CUDA_CALL(cuMemAlloc(&img->device_mem, img->width*img->height*3*4));
            img->rgb=(uint8_t *)img->device_mem;
            break;
        }
        case IMAGE_FORMAT_RGB_PLANAR_FP16_HOST:
        {
            CHECK_CUDA_CALL(cuMemAllocHost(&img->host_mem, img->width*img->height*3*2));
            img->rgb=(uint8_t *)img->host_mem;
            break;
        }
        case IMAGE_FORMAT_RGB_PLANAR_FP32_HOST:
        {
            CHECK_CUDA_CALL(cuMemAllocHost(&img->host_mem, img->width*img->height*3*4));
            img->rgb=(uint8_t *)img->host_mem;
            break;
        }
        default:
            assert(0);
            break;
    }
}

image_t *create_image_no_surface_memory(int width, int height, image_format_t fmt)
{
    image_t *img=(image_t *)malloc(sizeof(image_t));
    if (!img) return 0;
    memset(img, 0, sizeof(image_t));
    img->width=width;
    img->height=height;
    img->format=fmt;
    img->reference_count=1;
    img->stream=create_custream();
    return img;
}

image_t *create_image(int width, int height, image_format_t fmt)
{
    image_t *ret=create_image_no_surface_memory(width, height, fmt);
    allocate_image_surfaces(ret);
    return ret;
}

image_t *image_reference(image_t *img)
{
    if (img==0) return 0;
    __sync_fetch_and_add(&img->reference_count, 1);
    return img;
}

void destroy_image(image_t *img)
{
    if (!img) return;
    int reference_count=__sync_fetch_and_add(&img->reference_count, -1);
    if (reference_count>1) return;

    cuStreamSynchronize(img->stream);
    destroy_custream(img->stream);

    switch(img->format)
    {
        case IMAGE_FORMAT_YUV420_HOST:
        case IMAGE_FORMAT_RGB24_HOST:
        case IMAGE_FORMAT_RGB_PLANAR_FP16_HOST:
        case IMAGE_FORMAT_RGB_PLANAR_FP32_HOST:
        {
            if (img->host_mem) cuMemFreeHost(img->host_mem);
            img->host_mem=0;
            break;
        }
        case IMAGE_FORMAT_YUV420_DEVICE:
        case IMAGE_FORMAT_NV12_DEVICE:
        case IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE:
        case IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE:
        case IMAGE_FORMAT_RGB24_DEVICE:
        {
            if (img->device_mem) cuMemFree(img->device_mem);
            img->device_mem=0;
            break;
        }
        default:
            assert(0);
    }
    free(img);
}

void image_sync(image_t *img)
{
    if (!img) return;
    cuStreamSynchronize(img->stream);
}

void image_add_dependency(image_t *img, image_t *depends_on)
{
    cuda_stream_add_dependency(img->stream, depends_on->stream);
}

const char *image_format_name(image_format_t format)
{
    switch(format)
    {
        case IMAGE_FORMAT_YUV420_HOST: return "yuv420 host";
        case IMAGE_FORMAT_YUV420_DEVICE: return "yuv420 device";
        case IMAGE_FORMAT_NV12_DEVICE: return "nv12 device";
        case IMAGE_FORMAT_RGB24_HOST: return "rgb24 host";
        case IMAGE_FORMAT_RGB24_DEVICE: return "rgb24 device";
        case IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE: return "fp16 device";
        case IMAGE_FORMAT_RGB_PLANAR_FP16_HOST: return "fp16 host";
        case IMAGE_FORMAT_RGB_PLANAR_FP32_HOST: return "fp32 host";
        case IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE: return "fp32 device";
        default: break;
    }
    return "unknown";
}
