#ifndef __MISC_H
#define __MISC_H

#include <stddef.h>
#include <stdint.h>
#include "image.h"

double time_now_sec();
void hash_2d(const uint8_t *mem, int w, int h, int stride, uint32_t *dest);
uint32_t hash_u32(const uint32_t *mem, int num);
const char *ubon_cstuff_get_version(void);
void decode_file(const char *file, void *context, void (*callback)(void *context, image_t *img),
                 float framerate=0, bool (*stop_callback)(void *context)=0);

#endif
