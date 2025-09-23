#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <string.h>

#include "pcap_stuff.h"
#include "rtp_receiver.h"
#include "h26x_assembler.h"
#include "simple_decoder.h"
#include "pcap_decoder.h"
#include "sdp_parser.h"
#include "display.h"
#include "cuda_stuff.h"
#include <unistd.h>

#define MAX_SDP_SIZE 4096

typedef struct network_play_context
{
    rtp_receiver_t *rtp_receiver;
    h26x_assembler_t *h26x_assembler;
    simple_decoder_t *decoder;
    const char *write_debug_annexb_file;
} network_play_context_t;

static void decoder_frame_callback(void *context, image_t *img)
{
    display_image("video",img);
}

static void h26x_assembler_callback(void *context, const h26x_frame_descriptor_t *desc)
{
    network_play_context_t *c=(network_play_context_t *)context;
    //h26x_print_frame_summary(desc);
    simple_decoder_decode(c->decoder, desc->annexb_data, desc->annexb_length);

    if (c->write_debug_annexb_file!=0)
    {
        static FILE *f=0;
        if (!f) f=fopen(c->write_debug_annexb_file,"wb");
        fwrite(desc->annexb_data, 1, desc->annexb_length, f);
        fflush(f);
    }
}

static void rtp_receiver_callback(void *context, const rtp_packet_t *packet)
{
    static bool inited=false;
    if (!inited)
    {
        inited=true;
        cuda_thread_init();
    }

    network_play_context_t *c=(network_play_context_t *)context;
    h26x_assembler_process_rtp(c->h26x_assembler, packet);
}

static void rtp_callback(void *context, uint8_t *rtp_packet, size_t packet_length, double capture_time)
{
    network_play_context_t *c=(network_play_context_t *)context;
    rtp_receiver_add_packet(c->rtp_receiver, rtp_packet, (int)packet_length);
}

int main(int argc, char *argv[]) {
    init_cuda_stuff();
    image_init();

    network_play_context_t pc={0};

    /*
    example ffmpeh command to use with below SDP:
    ffmpeg -re -stream_loop -1 -i /mldata/video/bc2.264   -vcodec libx264 -preset veryfast -tune zerolatency   -g 30 -keyint_min 30 -bf 0 -profile:v baseline -pix_fmt yuv420p   -x264-params repeat_headers=1   -f rtp -payload_type 96   -srtp_out_suite AES_CM_128_HMAC_SHA1_80   -srtp_out_params MTIzNDU2Nzg5MDEyMzQ1Njc4OTAxMjM0NTY3ODkw   srtp://127.0.0.1:5004?pkt_size=1200
    */

    const char *sdp_str =
    "m=video 5004 RTP/AVP 96\r\n"
    "a=rtpmap:96 H265/90000\r\n"
    "a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:MTIzNDU2Nzg5MDEyMzQ1Njc4OTAxMjM0NTY3ODkw\r\n";

    //parsed_sdp_t *sdp=parse_sdp(sdp_str);
    //print_parsed_sdp(sdp);

    sdp_t sdp;
    pc.rtp_receiver=rtp_receiver_create((void *)&pc, rtp_receiver_callback);
    int ret=set_sdp(sdp_str, &sdp, pc.rtp_receiver);

    if (ret!=0) {
        log_fatal("Failed to parse SDP %d\n",ret);
    }

    pc.h26x_assembler=h26x_assembler_create(sdp.is_h265? H26X_CODEC_H265 : H26X_CODEC_H264,( void *)&pc, h26x_assembler_callback);
    pc.decoder = simple_decoder_create(( void *)&pc, decoder_frame_callback, sdp.is_h265 ? SIMPLE_DECODER_CODEC_H265 : SIMPLE_DECODER_CODEC_H264);
    //pc.write_debug_annexb_file="out.264";

    if (rtp_receiver_start_receive(pc.rtp_receiver, "0.0.0.0", (uint16_t)sdp.port) != 0)
    {
        log_fatal("Failed to bind UDP socket on port %d\n", sdp.port);
    }

    int cnt=0;
    while(1)
    {
        usleep(2000000);
        rtp_stats_t rtp_stats;
        h26x_nal_stats_t nal_stats;
        rtp_receiver_fill_stats(pc.rtp_receiver, &rtp_stats);
        h26x_assembler_fill_stats(pc.h26x_assembler, &nal_stats);
        print_rtp_stats(&rtp_stats);
        print_nal_stats(&nal_stats);
        cnt++;
        //if (cnt>=10) exit(0);
    }

    return 0;
}
