/*
git clone https://github.com/cisco/openh264.git
cd openh264
make && sudo make install
sudo ldconfig
*/
#include <string.h>
#include <wels/codec_api.h>
#include "simple_decoder.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "libyuv.h"
#include "image.h"

struct simple_decoder
{
    ISVCDecoder *decoder;
    SBufferInfo bufInfo;
    int out_width;
    int out_height;
    void *context;
    uint8_t *bitstream_buffer;
    int bitstream_buffer_fullness;
    int bitstream_buffer_size;
    void (*frame_callback)(void *context, image_t *decoded_frame);
};

static void openh264_trace_cb (void *ctx, int level, const char *string)
{
    printf("OPENH264 %s\n",string);
}


simple_decoder_t * __attribute__((weak)) simple_decoder_create(void *context, void (*frame_callback)(void *context, image_t *decoded_frame), simple_decoder_codec_t codec)
{
    simple_decoder_t *dec = (simple_decoder_t *)malloc(sizeof(simple_decoder_t));
    if (dec==0) return 0;

    ISVCDecoder *pSvcDecoder;
    SDecodingParam sDecParam = {0};

    WelsCreateDecoder(&dec->decoder);
    sDecParam.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_AVC;

    dec->decoder->Initialize(&sDecParam);

    if (0)
    {
        WelsTraceCallback log_cb = openh264_trace_cb;
        int log_level = WELS_LOG_DETAIL;//WELS_LOG_WARNING;
        dec->decoder->SetOption (DECODER_OPTION_TRACE_LEVEL, &log_level);
        dec->decoder->SetOption (DECODER_OPTION_TRACE_CALLBACK,(void *) &log_cb);
    }


    dec->out_width = 1280;
    dec->out_height = 720;

    dec->frame_callback=frame_callback;
    dec->context = context;
    dec->bitstream_buffer_size=512*1024;
    dec->bitstream_buffer=(uint8_t *)malloc(dec->bitstream_buffer_size);
    dec->bitstream_buffer_fullness=0;


    return dec;
}

void __attribute__((weak)) simple_decoder_destroy(simple_decoder_t *dec)
{
    if (dec)
    {
        dec->decoder->Uninitialize();
        free(dec->bitstream_buffer);
        WelsDestroyDecoder(dec->decoder);
        free(dec);
    }
}

int find_next_start_code(uint8_t* stream, int start, int size)
{
    for (int i = start; i < size - 3; i++)
    {
        if (stream[i] == 0x00 && stream[i+1] == 0x00 && stream[i+2] == 0x01)
        {
            return i;
        }
    }
    return -1;
}

static void simple_decoder_decode_one_nalu(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size)
{
    uint8_t *data[3] = {0};
    memset(&dec->bufInfo, 0, sizeof(SBufferInfo));
    DECODING_STATE state = dec->decoder->DecodeFrame2(bitstream_data, data_size, data, &dec->bufInfo);
    if (state == dsErrorFree && dec->bufInfo.iBufferStatus == 1)
    {
        // Perform bilinear scaling
        int src_width = dec->bufInfo.UsrData.sSystemBuffer.iWidth;
        int src_height = dec->bufInfo.UsrData.sSystemBuffer.iHeight;

        image_t *img=image_create_no_surface_memory(src_width, src_height, IMAGE_FORMAT_YUV420_HOST);

        img->y=data[0];
        img->u=data[1];
        img->v=data[2];
        img->stride_y=dec->bufInfo.UsrData.sSystemBuffer.iStride[0];
        img->stride_uv=dec->bufInfo.UsrData.sSystemBuffer.iStride[1];

        image_t *scaled_img=image_scale(img, dec->out_width, dec->out_height);

        dec->frame_callback(dec->context, scaled_img);
        image_destroy(scaled_img);
        image_destroy(img);
    }
}

void __attribute__((weak)) simple_decoder_decode(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size)
{
    assert(data_size+dec->bitstream_buffer_fullness<=dec->bitstream_buffer_size);
    memcpy(dec->bitstream_buffer+dec->bitstream_buffer_fullness, bitstream_data, data_size);
    dec->bitstream_buffer_fullness+=data_size;


    while(1)
    {
        int pos=find_next_start_code(dec->bitstream_buffer,3,dec->bitstream_buffer_fullness);
        if (pos==-1) return;
        if (dec->bitstream_buffer[pos-1]==0) pos--;
        int start=0;
        //if (dec->bitstream_buffer[0]==0 && dec->bitstream_buffer[1]==0 && dec->bitstream_buffer[2]==1) start=3;
       // if (dec->bitstream_buffer[0]==0 && dec->bitstream_buffer[1]==0 && dec->bitstream_buffer[2]==0 && dec->bitstream_buffer[3]==1) start=4;
        simple_decoder_decode_one_nalu(dec, dec->bitstream_buffer+start, pos-start);
        memmove(dec->bitstream_buffer, dec->bitstream_buffer+pos, dec->bitstream_buffer_fullness-pos);
        dec->bitstream_buffer_fullness-=pos;
    }
}

void __attribute__((weak)) simple_decoder_set_framerate(simple_decoder_t *dec, double fps)
{
    (void)dec; (void)fps;
    printf("%s:%d This feature is not yet implemented", __func__, __LINE__);
}

void __attribute__((weak)) simple_decoder_set_output_format(simple_decoder_t *dec, image_format_t fmt)
{
    (void)dec; (void)fmt;
    printf("%s:%d This feature is not yet implemented", __func__, __LINE__);
}

void __attribute__((weak)) simple_decoder_set_max_time(simple_decoder_t *dec, double max_time)
{
    (void)dec; (void)max_time;
    printf("%s:%d This feature is not yet implemented", __func__, __LINE__);
}
