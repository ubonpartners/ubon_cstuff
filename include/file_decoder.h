#ifndef __FILE_DECODER_H
#define __FILE_DECODER_H

#include "simple_decoder.h"

float file_decoder_parse_fps(const char *s);
simple_decoder_codec_t file_decoder_parse_codec(const char *file);
void decode_file(const char *file, void *context, void (*callback)(void *context, image_t *img),
                 float framerate=0, bool (*stop_callback)(void *context)=0);

#endif
