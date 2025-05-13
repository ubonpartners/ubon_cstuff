#ifndef __IMAGE_H
#define __IMAGE_H

typedef struct image image_t;

typedef enum image_format {
  IMAGE_FORMAT_NONE=0,
  IMAGE_FORMAT_YUV420_HOST,             // Currently assumed BT.709 limited range
  IMAGE_FORMAT_YUV420_DEVICE,           // Currently assumed BT.709 limited range
  IMAGE_FORMAT_NV12_DEVICE,
  IMAGE_FORMAT_RGB24_HOST,
  IMAGE_FORMAT_RGB24_DEVICE,
  IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE,
  IMAGE_FORMAT_RGB_PLANAR_FP16_HOST,
  IMAGE_FORMAT_RGB_PLANAR_FP32_HOST,
  IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE,
  IMAGE_FORMAT_MONO_HOST,               // YUV420 but Y-plane only
  IMAGE_FORMAT_MONO_DEVICE,             // YUV420 but Y-plane only
} image_format_t;

#define NUM_IMAGE_FORMATS 12

static bool image_format_is_device(image_format_t format)
{
  return   (format==IMAGE_FORMAT_YUV420_DEVICE)
         ||(format==IMAGE_FORMAT_NV12_DEVICE)
         ||(format==IMAGE_FORMAT_RGB24_DEVICE)
         ||(format==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE)
         ||(format==IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE)
         ||(format==IMAGE_FORMAT_MONO_DEVICE);
}

static bool image_format_is_host(image_format_t format)
{
  return ~image_format_is_device(format);
}

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
    image_t *referenced_surface;
};

// call before using any image functions
// does things like creates a table for doing format conversion
void image_init();
const char *image_format_name(image_format_t format);

// surfaces are reference counted and immutable, this allows long chains of dependent
// asynchronous operations to be automatically queued up, whereas the application
// doesn't have to care

// create image: allocate a new surface and it's video memory
image_t *create_image(int width, int height, image_format_t fmt);
// create_image_no_surface_memory: create an empty shell surface with
// no underlying video memory. This is useful in cases where different
// surfaces share underlying memory - e.g. if one is a crop of another
image_t *create_image_no_surface_memory(int width, int height, image_format_t fmt);
// destroy image reduces the ref count by one but the surface will remain
// until the ref count is zero.
void destroy_image(image_t *img);
image_t *image_reference(image_t *img);
// block & wait for all outstanding generating 'img' to be completed
void image_sync(image_t *img);
// mark that the generation of surface 'img' depends on the pixels
// of surface 'depends_on' so the operations should be asynchronously
// queued to run after
void image_add_dependency(image_t *img, image_t *depends_on);
// clear image - only used for debug, should ever need to clear an image
void clear_image(image_t *img);
// scale image to new size
image_t *image_scale(image_t *img, int width, int height);
// convert image to different format
// any format to any other should work although some are not direct
// and work via an intermediate format
image_t *image_convert(image_t *img, image_format_t format);
// generate a 32 bit value from a hash of the surface's pixel contents
// this is very useful for checking for reproducibility bugs
uint32_t image_hash(image_t *img);
// apply Gaussian blur to image
image_t *image_blur(image_t *img);
// returns a 1/4 size surface where each 'pixel' is mean absolute difference of the corresponding
// 4x4 block of the two input surfaces.
image_t *image_mad_4x4(image_t *a, image_t *b);
// image blend: product a new surface by copying a sub-rectange of src2 over src
// the sub-rectange is defined by sx,dy,w,h in src2 and is copied to position dx,dy
image_t *image_blend(image_t *src, image_t *src2, int sx, int sy, int w, int h, int dx, int dy);
// derive a new image from a crop of an existing one. Note in some cases this doesn't actually do
// any operations just uses reference counting and pointer manipulations so is 'free'
image_t *image_crop(image_t *img, int x, int y, int w, int h);
// pad an image by adding extra pixels to the edges, using the specified RGB colour
image_t *image_pad(image_t *img, int left, int top, int right, int bottom, uint32_t RGB);

#endif