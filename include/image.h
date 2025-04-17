#ifndef __IMAGE_H
#define __IMAGE_H

typedef struct image image_t;

typedef enum image_format {
  IMAGE_FORMAT_YUV420_HOST,
  IMAGE_FORMAT_YUV420_DEVICE,
  IMAGE_FORMAT_NV12_DEVICE,
  IMAGE_FORMAT_RGB24_HOST,
  IMAGE_FORMAT_RGB24_DEVICE,
  IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE,
  IMAGE_FORMAT_RGB_PLANAR_FP16_HOST,
  IMAGE_FORMAT_RGB_PLANAR_FP32_HOST,
  IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE,
} image_format_t;

#include <cuda.h>

struct image
{
    int width;
    int height;
    image_format_t format;
    uint8_t *y;
    uint8_t *u;
    uint8_t *v;
    uint8_t *rgb;
    volatile int reference_count;
    int stride_y, stride_uv, stride_rgb;
    CUstream stream; // any outstanding work on this surface will put a depency on this stream
    CUdeviceptr device_mem;
    void *host_mem;
    int device_mem_size;
    int host_mem_size;
};

const char *image_format_name(image_format_t format);

image_t *create_image(int width, int height, image_format_t fmt);
image_t *create_image_no_surface_memory(int width, int height, image_format_t fmt);
void destroy_image(image_t *img);
image_t *image_reference(image_t *img);
void image_sync(image_t *img); // wait for all outstanding ops on img 

void image_add_dependency(image_t *img, image_t *depends_on);

void clear_image(image_t *img);
image_t *image_scale(image_t *img, int width, int height);
image_t *image_convert(image_t *img, image_format_t format);
uint32_t image_hash(image_t *img);

#endif