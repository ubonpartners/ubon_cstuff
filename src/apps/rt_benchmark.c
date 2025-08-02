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
#include "yaml_stuff.h"

#define MAX_STREAMS     200

typedef struct rtp_packet
{
    int packet_length;
    double capture_time;
    uint8_t *data;
} rtp_packet_t;

#define MAX_PACKETS 30000

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

static std::string rt_benchmark(parsed_pcap_t **parsed, int n_parsed, int num_streams, double target_runtime)
{
    const char *config="/mldata/config/track/trackers/uc_reid.yaml";
    context_t ctx;
    memset(&ctx, 0, sizeof(context_t));

    ctx.tss = track_shared_state_create(config);
    for(int i=0;i<num_streams;i++)
    {
        ctx.ss[i].ts = track_stream_create(ctx.tss, &ctx.ss[i], track_result);
        ctx.ss[i].parsed_pcap=parsed[i%n_parsed];
        track_stream_set_sdp(ctx.ss[i].ts, ctx.ss[i].parsed_pcap->sdp);
        track_stream_set_minimum_frame_intervals(ctx.ss[i].ts, 1.0/10.0, 10.0);
    }

    infer_config_t inf_config={0};
    inf_config.limit_max_width = 640;
    inf_config.set_limit_max_width = true;
    inf_config.limit_max_height = 640;
    inf_config.set_limit_max_height = true;
    track_shared_state_configure_inference(ctx.tss, &inf_config);

    usleep(100000);

    double start_time=profile_time();
    double last_time=start_time;
    for(int i=0;i<num_streams;i++) ctx.ss[i].running=true;
    while(1)
    {
        if (profile_time()>start_time+target_runtime) break;
        usleep(10*1000);
        double time_now=profile_time();
        double delta=time_now-last_time;
        last_time+=delta;

        for(int i=0;i<num_streams;i++)
        {
            stream_state_t *ss=&ctx.ss[i];
            ss->packet_play_time+=delta;
            parsed_pcap_t *pp=ss->parsed_pcap;
            uint8_t *pkts[32];
            int lens[32];
            int n_add=0;
            while(1)
            {
                if (ss->packet_play_offs>=pp->num_packets)
                {
                    printf("RTP loop\n");
                    ss->packet_play_offs=0;
                    ss->packet_play_time=0;
                    break;
                }
                rtp_packet_t *pkt=pp->pkt[ss->packet_play_offs];
                if (pkt->capture_time<ss->packet_play_time)
                {
                    pkts[n_add]=pkt->data;
                    lens[n_add]=pkt->packet_length;
                    n_add++;
                    if (n_add>=32)
                    {
                        track_stream_add_rtp_packets(ss->ts, n_add, pkts, lens);
                        n_add=0;
                    }
                    ss->packet_play_offs++;
                    continue;
                }
                break;
            }
            if (n_add>0)
            {
                //printf("TS %p add %d\n",ss->ts,n_add);
                track_stream_add_rtp_packets(ss->ts, n_add, pkts, lens);
            }
        }
    }
    double runtime=profile_time()-start_time;
    for(int i=0;i<num_streams;i++) ctx.ss[i].running=false;

    float mean_latency_90=0;
    float mean_latency_50=0;
    float min_fps=1000.0;
    float max_fps=0;
    float total_fps=0;
    for(int i=0;i<num_streams;i++)
    {
        const char *stream_stats=track_stream_get_stats(ctx.ss[num_streams/2].ts);
        YAML::Node root=yaml_load(stream_stats);
        mean_latency_90+=root["main_processing"]["stats"]["pipeline_latency_histogram"]["centile_90"].as<float>();
        mean_latency_50+=root["main_processing"]["stats"]["pipeline_latency_histogram"]["centile_50"].as<float>();
        free((void*)stream_stats);

        stream_state_t *ss=&ctx.ss[i];
        float fps=ss->tracked_frames/runtime;
        min_fps=std::min(min_fps, fps);
        max_fps=std::max(max_fps, fps);
        total_fps+=fps;

    }
    mean_latency_90/=num_streams;
    mean_latency_50/=num_streams;
    float mean_fps=total_fps/num_streams;

    const char *stream_stats=track_stream_get_stats(ctx.ss[num_streams/2].ts);
    const char *shared_state_stats=track_shared_state_get_stats(ctx.tss);

    for(int i=0;i<num_streams;i++) track_stream_destroy(ctx.ss[i].ts);
    track_shared_state_destroy(ctx.tss);

    if (1) {
        const char *platform_stats=platform_get_stats();
        printf("======== PLATFORM STATS ===========\n");
        printf("%s\n",platform_stats);
        free((void*)platform_stats);
        printf("======== SHARED TRACK STATS ===========\n");
        printf("%s\n\n",shared_state_stats);
        free((void*)shared_state_stats);
        printf("======== STREAM STATS) ===========\n");
        printf("%s\n\n", stream_stats);
        free((void*)stream_stats);
    }

    std::ostringstream oss;
    oss << "Streams " << std::setw(4)  << num_streams
        << " FPS:"
        << " " << std::setw(5)  << std::fixed << std::setprecision(1) << min_fps
        << " - " << std::setw(5)  << std::fixed << std::setprecision(1) << max_fps
        << " Tot " << std::setw(6)  << std::fixed << std::setprecision(1) << total_fps
        << " Avg " << std::setw(5)  << std::fixed << std::setprecision(1) << mean_fps
        << " Lat50 " << std::setw(5)  << std::fixed << std::setprecision(3) << mean_latency_50
        << " Lat90 " << std::setw(5)  << std::fixed << std::setprecision(3) << mean_latency_90;

    return oss.str();
}


static const char *inputs[]={
    "/mldata/video/test/clip1_1280x720_5.00fps_h264.pcap",
    "/mldata/video/test/clip2_1280x720_5.00fps_h264.pcap",
    "/mldata/video/test/clip3_1280x720_5.00fps_h264.pcap"
};

int main(int argc, char *argv[]) {

    log_debug("ubon_cstuff version = %s", ubon_cstuff_get_version());
    init_cuda_stuff();
    log_debug("Init cuda done");
    log_debug("Initial GPU mem %f",get_process_gpu_mem(false, false));
    image_init();

    int n_in=sizeof(inputs)/sizeof(const char *);

    parsed_pcap_t *parsed[n_in];
    for(int i=0;i<n_in;i++) parsed[i]=parse_pcap(inputs[i]);

    std::ostringstream r;

    int min_str=(platform_is_jetson()) ? 4 : 14;
    int max_str=(platform_is_jetson()) ? 20 : 100;
    int step=(platform_is_jetson()) ? 2 : 10;
    double target_runtime=platform_is_jetson() ? 30 : 10;

    for(int ns=min_str;ns<=max_str;ns+=step)
    {
        r << rt_benchmark(parsed, n_in, ns, target_runtime) << "\n";
        std::cout << r.str();
    }
}
