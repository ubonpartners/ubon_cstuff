#include <unistd.h>
#include "misc.h"
#include "display.h"
#include "profile.h"
#include "cuda_stuff.h"
#include "h26x_assembler.h"
#include "file_decoder.h"
#include "mp4_writer.h"

static void cb(void *context, const h26x_frame_descriptor_t *desc)
{
    mp4_writer_t *wr=(mp4_writer_t *)context;
    mp4_writer_add_video_frame(wr, desc->extended_rtp_timestamp, desc->annexb_data, desc->annexb_length);
}

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();
    const char *infile=argv[1];

    float fps=file_decoder_parse_fps(infile);
    simple_decoder_codec_t codec=file_decoder_parse_codec(infile);

    mp4_writer_t *wr=mp4_writer_create("out1.mp4", (codec==SIMPLE_DECODER_CODEC_H265));
    h26x_assembler_t *a=h26x_assembler_create((codec==SIMPLE_DECODER_CODEC_H264) ? H26X_CODEC_H264 : H26X_CODEC_H265,
                                              wr,cb,true);

    printf("opening %s\n",argv[1]);
    FILE *f=fopen(argv[1], "rb");
    while(1)
    {
        uint8_t temp[2048];
        size_t r=fread(temp, 1, 2048, f);
        h26x_assembler_process_raw_video(a,
                                     fps,
                                     temp,
                                     (int)r);
        if (r<2048) break;
    }
    printf("done!");
    mp4_writer_destroy(wr);
    return 1;
}
