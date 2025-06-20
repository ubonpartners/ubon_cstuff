#ifndef __SIMPLE_DECODER_H
#define __SIMPLE_DECODER_H

#include <stdint.h>

typedef enum {
    SIMPLE_DECODER_CODEC_H264 = 0,
    SIMPLE_DECODER_CODEC_H265 = 1
} simple_decoder_codec_t;

typedef struct simple_decoder simple_decoder_t;

#include "image.h"

simple_decoder_t *simple_decoder_create(void *context, void (*frame_callback)(void *context, image_t *decoded_frame), simple_decoder_codec_t codec);
void simple_decoder_set_framerate(simple_decoder_t *dec, double fps);
void simple_decoder_destroy(simple_decoder_t *dec);
void simple_decoder_decode(simple_decoder_t *dec, uint8_t *data, int data_size);

#endif
