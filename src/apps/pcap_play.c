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
#include "display.h"
#include "cuda_stuff.h"

#define MAX_SDP_SIZE 4096

#define MAX_REORDER 16

typedef struct pcap_play_context
{
    rtp_receiver_t *rtp_receiver;
    h26x_assembler_t *h26x_assembler;
    simple_decoder_t *decoder;
    const char *write_debug_annexb_file;

    int reorder_percent;
    int drop_percent;

    uint8_t reordered_packets[MAX_REORDER][2048];
    int reordered_packet_lengths[MAX_REORDER];
    int num_reordered;
} pcap_play_context_t;

static void decoder_frame_callback(void *context, image_t *img)
{
    display_image("video",img);
}

static void h26x_assembler_callback(void *context, const h26x_frame_descriptor_t *desc)
{
    pcap_play_context_t *c=(pcap_play_context_t *)context;
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
    pcap_play_context_t *c=(pcap_play_context_t *)context;
    h26x_assembler_process_rtp(c->h26x_assembler, packet);
}

static void rtp_callback(void *context, uint8_t *rtp_packet, size_t packet_length, double capture_time)
{
    pcap_play_context_t *c=(pcap_play_context_t *)context;

    bool drop=(rand()&1023)<(10*c->drop_percent);
    if (drop) return;

    bool reorder=((rand()&1023)<(10*c->reorder_percent))
                ||((c->num_reordered>0) && ((rand()&1024)<512));

    if ((c->num_reordered<MAX_REORDER) && reorder)
    {
        memcpy(&c->reordered_packets[c->num_reordered][0], rtp_packet, packet_length);
        c->reordered_packet_lengths[c->num_reordered]=(int)packet_length;
        c->num_reordered++;
        return;
    }


    if (c->num_reordered>0)
    {
        for(int i=c->num_reordered-1;i>=0;i=i-1)
        {
            rtp_receiver_add_packet(c->rtp_receiver, c->reordered_packets[i], c->reordered_packet_lengths[i]);
        }
        c->num_reordered=0;
    }

    rtp_receiver_add_packet(c->rtp_receiver, rtp_packet, (int)packet_length);
}

int main(int argc, char *argv[]) {
    init_cuda_stuff();
    image_init();

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.pcap> <reorder_percent> <drop_percent>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];

    if (0)
    {
        pcap_decoder_t *dec=pcap_decoder_create(argv[1]);
        while(1)
        {
            image_t *i=pcap_decoder_get_frame(dec);
            if (i==0) exit(0);
            display_image("video",i);
            image_destroy(i);
        }
    }

    int drop_percent=0;
    int reorder_percent=0;

    if (argc>=3) reorder_percent=atoi(argv[2]);
    if (argc>=4) drop_percent=atoi(argv[3]);

    pcap_play_context_t pc={0};

    // Call parser
    char sdp[MAX_SDP_SIZE];
    if (parse_pcap_sdp(filename, sdp, sizeof(sdp))) {
        printf("Found SDP:\n%s\n", sdp);
    } else {
        printf("No SDP found.\n");
    }

    bool is_h265=strstr(sdp, "H265")!=0;
    if (is_h265)
        printf("Using H265\n");
    else
        printf("Using H264\n");

    printf("Reorder %d%% Drop %d%%\n",reorder_percent, drop_percent);

    pc.reorder_percent=reorder_percent;
    pc.drop_percent=drop_percent;
    pc.rtp_receiver=rtp_receiver_create((void *)&pc, rtp_receiver_callback);
    pc.h26x_assembler=h26x_assembler_create(is_h265? H26X_CODEC_H265 : H26X_CODEC_H264,( void *)&pc, h26x_assembler_callback);
    pc.decoder = simple_decoder_create(( void *)&pc, decoder_frame_callback, is_h265 ? SIMPLE_DECODER_CODEC_H265 : SIMPLE_DECODER_CODEC_H264);
    parse_pcap_rtp(filename, (void *)&pc, rtp_callback);


    rtp_stats_t rtp_stats;
    h26x_nal_stats_t nal_stats;
    rtp_receiver_fill_stats(pc.rtp_receiver, &rtp_stats);
    h26x_assembler_fill_stats(pc.h26x_assembler, &nal_stats);
    print_rtp_stats(&rtp_stats);
    print_nal_stats(&nal_stats);

    return 0;
}
