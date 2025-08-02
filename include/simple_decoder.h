#ifndef __SIMPLE_DECODER_H
#define __SIMPLE_DECODER_H

#include <stdint.h>
#include <yaml-cpp/yaml.h>

typedef enum {
    SIMPLE_DECODER_CODEC_UNKNOWN = 0,
    SIMPLE_DECODER_CODEC_H264 = 1,
    SIMPLE_DECODER_CODEC_H265 = 2
} simple_decoder_codec_t;

typedef struct simple_decoder simple_decoder_t;

#include "image.h"

simple_decoder_t *simple_decoder_create(void *context, void (*frame_callback)(void *context, image_t *decoded_frame),
                                        simple_decoder_codec_t codec, bool low_latency=true);
void simple_decoder_destroy(simple_decoder_t *dec);
void simple_decoder_decode(simple_decoder_t *dec, uint8_t *data, int data_size, double frame_time=-1000.0, bool force_skip=false);
void simple_decoder_set_output_format(simple_decoder_t *dec, image_format_t fmt);
void simple_decoder_constrain_output(simple_decoder_t *dec, int max_width, int max_height, double min_time_delta);
YAML::Node simple_decoder_get_stats(simple_decoder *dec);
#endif
