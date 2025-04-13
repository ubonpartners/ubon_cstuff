#ifndef __JPEG_H
#define __JPEG_H

#include "image.h"

image_t *decode_jpeg(uint8_t *buffer, size_t size);
image_t *load_jpeg(const char *file);
void save_jpeg(const char *filename, image_t *img);

#endif
