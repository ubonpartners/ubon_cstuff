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

    bool john_format=false;

    float fps=file_decoder_parse_fps(infile);
    simple_decoder_codec_t codec=file_decoder_parse_codec(infile);
    if (codec==SIMPLE_DECODER_CODEC_UNKNOWN)
    {
        log_warn("Assuming file is in John's NALU format, with H264");
        codec=SIMPLE_DECODER_CODEC_H264;
        john_format=true;
    }
    else
    {
        log_warn("Assuming file is Annex-B format, detected codec %s, %ffps ",
            (codec==SIMPLE_DECODER_CODEC_H264) ? "H264" : "H265", fps);
    }

    const char *out="out.mp4";
    if (argc>2) out=argv[2];

    mp4_writer_t *wr=mp4_writer_create(out, (codec==SIMPLE_DECODER_CODEC_H265));
    h26x_assembler_t *a=h26x_assembler_create((codec==SIMPLE_DECODER_CODEC_H264) ? H26X_CODEC_H264 : H26X_CODEC_H265,
                                              wr,cb,true);

    log_info("opening %s",argv[1]);

    FILE *f=fopen(argv[1], "rb");

    if (john_format==false)
    {
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
    }
    else
    {
        int nalu_buf_size=1024*1024;
        uint8_t *nalu_buf=(uint8_t *)malloc(nalu_buf_size);
        uint64_t first_timestamp=0;
        bool first_nalu=true;
        while(1)
        {
            uint64_t ts;
            uint32_t len;
            if (8!=fread(&ts, 1, 8, f)) break;
            if (4!=fread(&len, 1, 4, f)) break;
            len=__builtin_bswap32(len);
            ts=__builtin_bswap64(ts);

            assert(len<1024*1024);
            if (len!=(int)fread(nalu_buf+4, 1, len, f)) break;

            if (ts==0) continue; // HACK : ignore initial extra TS=0 packets
            if (first_nalu)
            {
                first_nalu=false;
                first_timestamp=ts;
            }
            double time=(ts-first_timestamp)/90000.0;

            nalu_buf[0]=(len>>24)&0xff;
            nalu_buf[1]=(len>>16)&0xff;
            nalu_buf[2]=(len>>8)&0xff;
            nalu_buf[3]=(len>>0)&0xff;
            (void)h26x_assembler_process_nalus(a, nalu_buf, len+4, time);
        }
    }
    log_info("all done, written %s", out);
    mp4_writer_destroy(wr);
    return 1;
}
