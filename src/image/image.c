#include <stdint.h>
#include <malloc.h>
#include <string.h>
#include "image.h"
#include "cuda_stuff.h"
#include <cuda.h>
#include <assert.h>
#include <mutex>
#include <stdint.h>
#include "misc.h"
#include "memory_stuff.h"

static bool async_cuda_mem=true;
static bool pinned_host_mem=false; // true is very very slow
static std::once_flag initFlag;
static bool image_inited=false;

static allocation_tracker_t image_alloc_device_tracker;
static allocation_tracker_t image_alloc_host_tracker;
static block_allocator_t *image_allocator=0;

static void image_mem_init()
{
    allocation_tracker_register(&image_alloc_device_tracker, "image device alloc");
    allocation_tracker_register(&image_alloc_host_tracker, "image host host");
    image_allocator=block_allocator_create("image allocator", sizeof(image_t));
}

static void allocate_image_device_mem(image_t *img, int size)
{
    img->device_mem_size=size;
    track_alloc(&image_alloc_device_tracker, size);

    if (async_cuda_mem)
    {
        img->device_mem=cuda_malloc_async((size_t)size, img->stream);
    }
    else
    {
        img->device_mem=cuda_malloc((size_t)size);
    }
}

static void allocate_image_host_mem(image_t *img, int size)
{
    track_alloc(&image_alloc_host_tracker, size);
    img->host_mem_size=size;
    if (pinned_host_mem)
        img->host_mem=cuda_malloc_host(size);
    else
        img->host_mem=malloc(size);
}

static void free_image_mem(image_t *img)
{
    if (img->host_mem)
    {
        cudaStreamSynchronize(img->stream);
        if (pinned_host_mem)
            cuda_free_host(img->host_mem);
        else
            free(img->host_mem);
        track_free(&image_alloc_host_tracker, img->host_mem_size);
    }
    if (img->device_mem)
    {
        if (async_cuda_mem)
        {
            cuda_free_async((void*)img->device_mem, img->stream);
        }
        else
        {
            cudaStreamSynchronize(img->stream);
            cuda_free(img->device_mem);
        }
        track_free(&image_alloc_device_tracker, img->device_mem_size);
    }
    img->host_mem=0;
    img->device_mem=0;
}

static void allocate_image_surfaces(image_t *img)
{
    if (!img) return;
    switch(img->format)
    {
        case IMAGE_FORMAT_YUV420_HOST:
        {
            int round=(img->width+31)&(~31);
            allocate_image_host_mem(img, round*img->height*3/2);
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
            allocate_image_host_mem(img, round*img->height*3);
            img->rgb=(uint8_t *)img->host_mem;
            img->stride_rgb=img->width*3;
            break;
        }
        case IMAGE_FORMAT_RGB24_DEVICE:
        {
            int round=(img->width+31)&(~31);
            allocate_image_device_mem(img, round*img->height*3);
            img->rgb=(uint8_t *)img->device_mem;
            img->stride_rgb=img->width*3;
            break;
        }
        case IMAGE_FORMAT_YUV420_DEVICE:
        {
            int round_w=(img->width+127)&(~127);
            int round_h=((img->height+15)&(~15));
            allocate_image_device_mem(img, (round_w*round_h*3)/2);
            img->y=(uint8_t *)img->device_mem;
            img->u=img->y+(round_w*round_h);
            img->v=img->u+((round_w*round_h)>>2);
            img->stride_y=round_w;
            img->stride_uv=round_w>>1;
            break;
        }
        case IMAGE_FORMAT_NV12_DEVICE:
        {
            int round=(img->width+31)&(~31);
            allocate_image_device_mem(img, round*img->height*3/2);
            img->stride_y=round;
            img->stride_uv=round;
            img->y=(uint8_t *)img->device_mem;
            img->u=img->y+(img->stride_y*img->height);
            img->v=img->u+1;
            break;
        }
        case IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE:
        {
            allocate_image_device_mem(img, img->width*img->height*3*2);
            img->rgb=(uint8_t *)img->device_mem;
            img->stride_rgb=img->width*img->height; // in elements
            break;
        }
        case IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE:
        {
            allocate_image_device_mem(img, img->width*img->height*3*4);
            img->rgb=(uint8_t *)img->device_mem;
            img->stride_rgb=img->width*img->height; // in elements
            break;
        }
        case IMAGE_FORMAT_RGB_PLANAR_FP16_HOST:
        {
            allocate_image_host_mem(img, img->width*img->height*3*2);
            img->rgb=(uint8_t *)img->host_mem;
            break;
        }
        case IMAGE_FORMAT_RGB_PLANAR_FP32_HOST:
        {
            allocate_image_host_mem(img, img->width*img->height*3*4);
            img->rgb=(uint8_t *)img->host_mem;
            break;
        }
        case IMAGE_FORMAT_MONO_DEVICE:
        {
            int round=(img->width+15)&(~15);
            allocate_image_device_mem(img, round*img->height);
            img->y=(uint8_t *)img->device_mem;
            img->stride_y=round;
            break;
        }
        case IMAGE_FORMAT_MONO_HOST:
        {
            int round=(img->width+31)&(~31);
            allocate_image_host_mem(img, round*img->height);
            img->y=(uint8_t *)img->host_mem;
            img->stride_y=round;
            break;
        }
        default:
            assert(0);
            break;
    }
}

static void image_free_callback(void *context, void *block)
{
    image_t *img=(image_t *)block;
    image_t *referenced_surface=img->referenced_surface;
    free_image_mem(img);
    destroy_cuda_stream(img->stream);
    if (referenced_surface) destroy_image(referenced_surface);
}

image_t *create_image_no_surface_memory(int width, int height, image_format_t fmt)
{
    image_t *img=(image_t *)block_alloc(image_allocator);
    if (!img) return 0;
    memset(img, 0, sizeof(image_t));
    img->width=width;
    img->height=height;
    img->format=fmt;
    img->stream=create_cuda_stream();
    block_set_free_callback(img, 0, image_free_callback);
    return img;
}

image_t *create_image(int width, int height, image_format_t fmt)
{
    assert(image_inited);
    image_t *ret=create_image_no_surface_memory(width, height, fmt);
    allocate_image_surfaces(ret);
    return ret;
}

image_t *image_reference(image_t *img)
{
    return (image_t*)block_reference(img);
}

void destroy_image(image_t *img)
{
    block_free(img);
}

void image_sync(image_t *img)
{
    if (!img) return;
    block_check(img);
    //log_debug("synchronize %dx%d %s",img->width,img->height,image_format_name(img->format));
    cudaStreamSynchronize(img->stream);
}

void image_add_dependency(image_t *img, image_t *depends_on)
{
    block_check(img);
    block_check(depends_on);
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
        case IMAGE_FORMAT_MONO_HOST: return "mono host";
        case IMAGE_FORMAT_MONO_DEVICE: return "mono device";
        default: break;
    }
    return "unknown";
}

extern void image_conversion_init();
static void do_image_init()
{
    image_conversion_init();
    image_mem_init();
    image_inited=true;
}

void image_init()
{
    std::call_once(initFlag, do_image_init);
}