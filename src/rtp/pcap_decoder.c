#include <pcap.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include "pcap_stuff.h"
#include "log.h"
#include "image.h"
#include "simple_decoder.h"
#include "h26x_assembler.h"
#include "rtp_receiver.h"
#include "pcap_decoder.h"

#define MAX_SDP_SIZE        512
#define MAX_DECODED_FRAMES  16

struct pcap_decoder
{
    char errbuf[PCAP_ERRBUF_SIZE];
    bool is_h265;
    int num_decoded_frames;
    image_t *decoded_frames[MAX_DECODED_FRAMES];
    pcap_t *pcap_handle;
    rtp_receiver_t *rtp_receiver;
    h26x_assembler_t *h26x_assembler;
    simple_decoder_t *decoder;
};

static void decoder_frame_callback(void *context, image_t *img)
{
    pcap_decoder_t *p=(pcap_decoder_t *)context;
    assert(p->num_decoded_frames<MAX_DECODED_FRAMES);
    p->decoded_frames[p->num_decoded_frames++]=image_reference(img);
}

static void h26x_assembler_callback(void *context, const h26x_frame_descriptor_t *desc)
{
    pcap_decoder_t *p=(pcap_decoder_t *)context;
    simple_decoder_decode(p->decoder, desc->annexb_data, desc->annexb_length, desc->extended_rtp_timestamp/90000.0);
}

static void rtp_receiver_callback(void *context, const rtp_packet_t *packet)
{
    pcap_decoder_t *p=(pcap_decoder_t *)context;
    //for(int i=0;i<12;i++) printf("%2.2x ",packet->data[i]);
    //printf("\n");
    h26x_assembler_process_rtp(p->h26x_assembler, packet);
}

pcap_decoder_t *pcap_decoder_create(const char *filename)
{
    pcap_decoder_t *p=(pcap_decoder_t *)malloc(sizeof(pcap_decoder_t));
    memset(p, 0, sizeof(pcap_decoder_t));

    p->pcap_handle = pcap_open_offline_with_tstamp_precision(filename, PCAP_TSTAMP_PRECISION_MICRO, p->errbuf);
    if (!p->pcap_handle)
    {
        log_error("Could not open file %s",filename);
        return 0;
    }

    char sdp[MAX_SDP_SIZE];
    if (parse_pcap_sdp(filename, sdp, sizeof(sdp))) {
        printf("Found SDP:\n%s\n", sdp);
    } else {
        printf("No SDP found.\n");
    }

    p->is_h265=strstr(sdp, "H265")!=0;
    if (p->is_h265)
        printf("Using H265\n");
    else
        printf("Using H264\n");

    p->rtp_receiver=rtp_receiver_create((void *)p, rtp_receiver_callback);
    p->h26x_assembler=h26x_assembler_create(p->is_h265? H26X_CODEC_H265 : H26X_CODEC_H264,( void *)p, h26x_assembler_callback);
    p->decoder = simple_decoder_create(p, decoder_frame_callback, p->is_h265 ? SIMPLE_DECODER_CODEC_H265 : SIMPLE_DECODER_CODEC_H264);
    return p;
}
void pcap_decoder_destroy(pcap_decoder_t *p)
{
    if (p)
    {
        if (p->pcap_handle) pcap_close(p->pcap_handle);
        if (p->decoder) simple_decoder_destroy(p->decoder);
        if (p->h26x_assembler) h26x_assembler_destroy(p->h26x_assembler);
        if (p->rtp_receiver) rtp_receiver_destroy(p->rtp_receiver);
        for(int i=0;i<p->num_decoded_frames;i++) image_destroy(p->decoded_frames[i]);
        free(p);
    }
}

image_t *pcap_decoder_get_frame(pcap_decoder_t *p)
{
    struct pcap_pkthdr *header;
    const u_char *packet;
    int res;

    while ((p->num_decoded_frames==0) && (res = pcap_next_ex(p->pcap_handle, &header, &packet)) >= 0) {
        if (res == 0) continue;

        const struct ip *ip_hdr = (struct ip *)(packet + 14);
        if (ip_hdr->ip_p != IPPROTO_UDP) continue;

        size_t ip_hdr_len = ip_hdr->ip_hl * 4;
        const struct udphdr *udp_hdr = (const struct udphdr *)((const uint8_t *)ip_hdr + ip_hdr_len);

        size_t udp_payload_len = ntohs(udp_hdr->uh_ulen) - sizeof(struct udphdr);
        const uint8_t *udp_payload = (const uint8_t *)udp_hdr + sizeof(struct udphdr);

        if (udp_payload_len >= 12 && ((udp_payload[0] & 0xC0) == 0x80)) {
            //double capture_time = header->ts.tv_sec + header->ts.tv_usec / 1e6;
            rtp_receiver_add_packet(p->rtp_receiver, (uint8_t *)udp_payload, udp_payload_len);
        }
    }

    image_t *ret=0;
    if (p->num_decoded_frames>0)
    {
        ret=p->decoded_frames[0];
        p->num_decoded_frames--;
        for(int i=0;i<p->num_decoded_frames;i++)
        {
            p->decoded_frames[i]=p->decoded_frames[i+1];
        }
    }
    return ret;
}