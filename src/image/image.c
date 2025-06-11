#include <stdint.h>
#include <malloc.h>
#include <string.h>
#include "image.h"
#include "cuda_stuff.h"
#include <cuda.h>
#include <assert.h>
#include <mutex>
#include <stdint.h>

static bool async_cuda_mem=true;
static std::once_flag initFlag;
static bool image_inited=false;

typedef struct image_mem_stats
{
    uint64_t outstanding_device_mem;
    uint64_t outstanding_host_mem;
    uint64_t outstanding_device_allocations;
    uint64_t outstanding_host_allocations;
    uint64_t total_device_mem;
    uint64_t total_host_mem;
    uint64_t total_device_allocations;
    uint64_t total_host_allocations;
    uint64_t hwm_host_mem;
    uint64_t hwm_host_allocations;
    uint64_t hwm_device_mem;
    uint64_t hwm_device_allocations;
} image_mem_stats_t;

static image_mem_stats_t memstats={};

void print_image_mem_stats(const image_mem_stats_t *stats)
{
    printf("Image Memory Stats:\n");
    printf("    HWM:\n");
    printf("        Device Memory (bytes):         %lu\n", stats->hwm_device_mem);
    printf("        Host Memory (bytes):           %lu\n", stats->hwm_host_mem);
    printf("        Device Allocations:            %lu\n", stats->hwm_device_allocations);
    printf("        Host Allocations:              %lu\n", stats->hwm_host_allocations);
    printf("    Outstanding:\n");
    printf("        Device Memory (bytes):         %lu\n", stats->outstanding_device_mem);
    printf("        Host Memory (bytes):           %lu\n", stats->outstanding_host_mem);
    printf("        Device Allocations:            %lu\n", stats->outstanding_device_allocations);
    printf("        Host Allocations:              %lu\n", stats->outstanding_host_allocations);
    printf("    Total:\n");
    printf("        Device Memory (bytes):         %lu\n", stats->total_device_mem);
    printf("        Host Memory (bytes):           %lu\n", stats->total_host_mem);
    printf("        Device Allocations:            %lu\n", stats->total_device_allocations);
    printf("        Host Allocations:              %lu\n", stats->total_host_allocations);
}

static void image_mem_init()
{
    // some stuff
}

static void atomic_max_u64(uint64_t *ptr, uint64_t val)
{
    uint64_t old = __atomic_load_n(ptr, __ATOMIC_RELAXED);
    while (val > old &&
           !__atomic_compare_exchange_n(ptr, &old, val,
                                        /*weak=*/0,
                                        __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED)) {
        // old is updated automatically
    }
}

static void allocate_image_device_mem(image_t *img, int size)
{
    img->device_mem_size=size;
    __atomic_add_fetch(&memstats.outstanding_device_mem, size, __ATOMIC_RELAXED);
    __atomic_add_fetch(&memstats.outstanding_device_allocations, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&memstats.total_device_mem, size, __ATOMIC_RELAXED);
    __atomic_add_fetch(&memstats.total_device_allocations, 1, __ATOMIC_RELAXED);
    atomic_max_u64(&memstats.hwm_device_mem, memstats.outstanding_device_mem);
    atomic_max_u64(&memstats.hwm_device_allocations, memstats.outstanding_device_allocations);
    if (async_cuda_mem)
    {
        CHECK_CUDART_CALL(cudaMallocAsync((void**)&img->device_mem, (size_t)size, img->stream));
    }
    else
    {
        CHECK_CUDART_CALL(cudaMalloc((void**)&img->device_mem, size));
    }
}

static void allocate_image_host_mem(image_t *img, int size)
{
    img->host_mem_size=size;
    __atomic_add_fetch(&memstats.outstanding_host_mem, size, __ATOMIC_RELAXED);
    __atomic_add_fetch(&memstats.outstanding_host_allocations, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&memstats.total_host_mem, size, __ATOMIC_RELAXED);
    __atomic_add_fetch(&memstats.total_host_allocations, 1, __ATOMIC_RELAXED);
    atomic_max_u64(&memstats.hwm_host_mem, memstats.outstanding_host_mem);
    atomic_max_u64(&memstats.hwm_host_allocations, memstats.outstanding_host_allocations);
    //CHECK_CUDART_CALL(cudaMallocHost(&img->host_mem, size));
    img->host_mem=malloc(size);
    //print_image_mem_stats(&memstats);
}

static void free_image_mem(image_t *img)
{
    if (img->host_mem)
    {
        cudaStreamSynchronize(img->stream);
        free(img->host_mem);
        //cudaFreeHost(img->host_mem);
        __atomic_sub_fetch(&memstats.outstanding_host_mem, img->host_mem_size, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&memstats.outstanding_host_allocations, 1, __ATOMIC_RELAXED);
    }
    if (img->device_mem)
    {
        if (async_cuda_mem)
        {
            cudaFreeAsync((void*)img->device_mem, img->stream);
        }
        else
        {
            cudaStreamSynchronize(img->stream);
            cudaFree((void*)img->device_mem);
        }
        __atomic_sub_fetch(&memstats.outstanding_device_mem, img->device_mem_size, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&memstats.outstanding_device_allocations, 1, __ATOMIC_RELAXED);
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
            int round_w=(img->width+63)&(~63);
            int round_h=((img->height+7)&(~7));
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

image_t *create_image_no_surface_memory(int width, int height, image_format_t fmt)
{
    image_t *img=(image_t *)malloc(sizeof(image_t));
    if (!img) return 0;
    memset(img, 0, sizeof(image_t));
    img->width=width;
    img->height=height;
    img->format=fmt;
    img->reference_count=1;
    img->stream=create_cuda_stream();
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
    if (img==0) return 0;
    __sync_fetch_and_add(&img->reference_count, 1);
    return img;
}

void destroy_image(image_t *img)
{
    if (!img) return;
    int reference_count=__sync_fetch_and_add(&img->reference_count, -1);
    if (reference_count>1) return;
    image_t *referenced_surface=img->referenced_surface;
    free_image_mem(img);
    destroy_cuda_stream(img->stream);
    free(img);
    if (referenced_surface) destroy_image(referenced_surface);
}

void image_sync(image_t *img)
{
    if (!img) return;
    assert(img->reference_count>0);
    //log_debug("synchronize %dx%d %s",img->width,img->height,image_format_name(img->format));
    cudaStreamSynchronize(img->stream);
}

void image_add_dependency(image_t *img, image_t *depends_on)
{
    assert(img->reference_count>0);
    assert(depends_on->reference_count>0);
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