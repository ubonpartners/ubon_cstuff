#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>  // for std::setw and std::setprecision
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#include "cuda_stuff.h"
#include "simple_decoder.h"
#include "display.h"
#include "track.h"
#include "misc.h"
#include "log.h"
#include "profile.h"
#include "memory_stuff.h"
#include "infer_thread.h"
#include "display.h"
#include "platform_stuff.h"
#include "pcap_stuff.h"

#define MAX_STREAMS     8

typedef struct rtp_packet
{
    int packet_length;
    double capture_time;
    uint8_t *data;
} rtp_packet_t;

#define MAX_PACKETS 16384

typedef struct parsed_pcap
{
    char sdp[256];
    int num_packets;
    rtp_packet_t *pkt[MAX_PACKETS];
} parsed_pcap_t;


typedef struct stream_state
{
    parsed_pcap_t *parsed_pcap;
    track_stream_t *ts;
    int packet_play_offs;
    double packet_play_time;
    bool running;
    uint32_t tracked_frames;
    uint32_t tracked_frames_nonskip;
    uint32_t face_embeddings;
    double time;
} stream_state_t;

typedef struct context
{
    track_shared_state_t *tss;
    stream_state_t ss[MAX_STREAMS];
} context_t;

static void track_result(void *context, track_results_t *r) {
    stream_state_t *ss = (stream_state_t *)context;
    if (ss->running)
    {
        if (r->result_type!=TRACK_FRAME_SKIP_FRAMERATE) ss->tracked_frames++;
        if (r->result_type == TRACK_FRAME_TRACKED_ROI || r->result_type == TRACK_FRAME_TRACKED_FULL_REFRESH) {
            ss->tracked_frames_nonskip++;
        }
        ss->time=r->time;
        detection_list_t *track_dets=r->track_dets;
        if (track_dets)
        {
            for(int i=0;i<track_dets->num_detections;i++)
            {
                detection_t *det=track_dets->det[i];
                if (det->face_embedding)
                {
                    if (embedding_get_time(det->face_embedding)==track_dets->time) ss->face_embeddings++;
                }
            }
        }
    }
}

static void rtp_callback(void *context, uint8_t *rtp_packet, size_t packet_length, double capture_time)
{
    parsed_pcap_t *p=(parsed_pcap_t *)context;
    rtp_packet_t *pkt=(rtp_packet_t *)malloc(sizeof(rtp_packet_t)+packet_length);
    pkt->capture_time=capture_time;
    pkt->packet_length=packet_length;
    pkt->data=(uint8_t *)(pkt+1);
    memcpy(pkt->data, rtp_packet, packet_length);
    assert(p->num_packets<MAX_PACKETS);
    p->pkt[p->num_packets++]=pkt;
}

static parsed_pcap_t *parse_pcap(const char *pcap)
{
    parsed_pcap_t *p=(parsed_pcap_t *)malloc(sizeof(parsed_pcap_t));
    memset(p, 0, sizeof(parsed_pcap_t));
    assert(true==parse_pcap_sdp(pcap, p->sdp, 256));
    parse_pcap_rtp(pcap, p, rtp_callback);
    double start=p->pkt[0]->capture_time;
    for(int i=0;i<p->num_packets;i++) p->pkt[i]->capture_time-=start;
    return p;
}

static void rt_benchmark()
{
    const char *config="/mldata/config/track/trackers/uc_reid.yaml";

    //parsed_pcap_t *parsed=parse_pcap("/mldata/video/operahouse.pcap");
    parsed_pcap_t *parsed=parse_pcap("/mldata/video/test/ind_off_1280x720_7.5fps_264.pcap");

    context_t ctx;
    memset(&ctx, 0, sizeof(context_t));

    int num_streams=MAX_STREAMS;

    //printf("SDP %s\n",parsed->sdp);

    ctx.tss = track_shared_state_create(config);
    for(int i=0;i<MAX_STREAMS;i++)
    {
        ctx.ss[i].ts = track_stream_create(ctx.tss, &ctx.ss[i], track_result);
        ctx.ss[i].parsed_pcap=parsed;
        track_stream_set_sdp(ctx.ss[i].ts, ctx.ss[i].parsed_pcap->sdp);
        track_stream_set_minimum_frame_intervals(ctx.ss[i].ts, 1.0/8.0, 10.0);
    }

    usleep(100000);

    double start_time=profile_time();
    double last_time=start_time;
    for(int i=0;i<num_streams;i++) ctx.ss[i].running=true;
    while(1)
    {
        usleep(10*1000);
        double time_now=profile_time();
        if (time_now>start_time+20.0) break;
        double delta=time_now-last_time;
        last_time+=delta;

        for(int i=0;i<num_streams;i++)
        {
            stream_state_t *ss=&ctx.ss[i];
            ss->packet_play_time+=delta;
            parsed_pcap_t *pp=ss->parsed_pcap;
            while(1)
            {
                if (ss->packet_play_offs>=pp->num_packets)
                {
                    ss->packet_play_offs=0;
                    ss->packet_play_time=0;
                    break;
                }
                rtp_packet_t *pkt=pp->pkt[ss->packet_play_offs];
                if (pkt->capture_time<ss->packet_play_time)
                {
                    //printf("play %d (%d) : \n",ss->packet_play_offs,pkt->packet_length);
                    //for(int i=0;i<12;i++) printf("%2.2x ",pkt->data[i]);
                    //printf("\n");
                    track_stream_add_rtp_packets(ss->ts, 1, &pkt->data, &pkt->packet_length);
                    ss->packet_play_offs++;
                    continue;
                }
                break;
            }
        }
    }
    double runtime=profile_time()-start_time;
    for(int i=0;i<num_streams;i++) ctx.ss[i].running=false;


    for(int i=0;i<num_streams;i++) track_stream_destroy(ctx.ss[i].ts);
    track_shared_state_destroy(ctx.tss);
    
    for(int i=0;i<num_streams;i++)
    {
        stream_state_t *ss=&ctx.ss[i];
        float fps=ss->tracked_frames/runtime;
        printf("Stream %2d/%2d : %f fr=%d %f\n",i,num_streams,ss->time,ss->tracked_frames,fps);
    }
}

int main(int argc, char *argv[]) {

    log_debug("ubon_cstuff version = %s", ubon_cstuff_get_version());
    init_cuda_stuff();
    log_debug("Init cuda done");
    log_debug("Initial GPU mem %f",get_process_gpu_mem(false, false));
    image_init();

    rt_benchmark();
}
