#ifndef __JPEG_THREAD_H
#define __JPEG_THREAD_H

typedef struct jpeg_thread jpeg_thread_t;
typedef struct jpeg jpeg_t;

#include "image.h"
#include "roi.h"


jpeg_thread_t *jpeg_thread_create(const char *yaml);
void jpeg_thread_destroy(jpeg_thread_t *t);

jpeg_t *jpeg_thread_encode(jpeg_thread_t *jt, image_t *img, roi_t roi, int max_w, int max_h, float quality=0, int encode_quality=0);
jpeg_t *jpeg_reference(jpeg_t *jpeg);
uint8_t *jpeg_get_data(jpeg_t *jpeg, size_t *ret_size);
double jpeg_get_time(jpeg_t *jpeg);
float jpeg_get_quality(jpeg_t *jpeg);
void jpeg_destroy(jpeg_t *jpeg);

#endif
