
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include "simple_decoder.h"
#include "cuda_stuff.h"

#if (UBONCSTUFF_PLATFORM == 1) // Orin Nano
struct simple_decoder
{
    int width;
    int height;
    int out_width;
    int out_height;
    void *context;
};

simple_decoder_t *simple_decoder_create(void *context, void (*frame_callback)(void *context, image_t *decoded_frame))
{
    simple_decoder_t *dec = NULL;

    (void)context; (void)frame_callback;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return dec;
}

void simple_decoder_destroy(simple_decoder_t *dec)
{
    (void)dec;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return;
}

void simple_decoder_decode(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size)
{
    (void)dec; (void)bitstream_data; (void)data_size;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return;
}
#endif //(UBONCSTUFF_PLATFORM == 1)

