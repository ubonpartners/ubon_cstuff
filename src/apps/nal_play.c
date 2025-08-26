
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cuda_stuff.h"
#include "simple_decoder.h"
#include "display.h"
#include "webcam.h"
#include "track.h"
#include "misc.h"
#include "profile.h"
#include "file_decoder.h"
#include "jpeg.h"
#include "yaml_stuff.h"
#include "default_setup.h"

typedef struct state
{
    track_shared_state_t *tss;
    track_stream_t *ts;
    image_t *img;
    double start_time;
} state_t;

static bool realtime=false;

static void track_result(void *context, track_results_t *r)
{
    state_t *s=(state_t *)context;
    printf("track_result: result type %d time %f, %d detections\n",r->result_type,r->time, (r->track_dets==0) ? 0 : r->track_dets->num_detections);
    if (r->track_dets!=0)
    {
        //detection_list_show(r->track_dets);
        // decode and display "frame jpeg"
        image_t *img=0;
        if (r->track_dets->frame_jpeg!=0)
        {
            size_t jpeg_data_length;
            uint8_t *jpeg_data=jpeg_get_data(r->track_dets->frame_jpeg, &jpeg_data_length);
            img=decode_jpeg(jpeg_data, (int)jpeg_data_length);
        }
        if (img!=0)
        {
            image_t *out_frame_rgb=detection_list_draw(r->track_dets, img);
            if (out_frame_rgb!=0)
            {
                display_image("video", out_frame_rgb);
                image_destroy(out_frame_rgb);
            }
            image_destroy(img);
        }
    }
}

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();

    state_t s;
    memset(&s, 0, sizeof(state_t));

    // modify default config so we get 720p30 'frame jpegs'
    // just so we can display these

    const char *basic="main_jpeg:\n"
                       "    enabled: true\n"
                       "    max_width: 1280\n"
                       "    max_height: 720\n"
                       "    min_interval_seconds: 0.03\n";

    const char *config=yaml_merge_string(DEFAULT_TRACKER_YAML, basic);

    s.tss=track_shared_state_create(config);
    s.ts=track_stream_create(s.tss, &s, track_result);
    s.start_time=profile_time();
    track_stream_set_minimum_frame_intervals(s.ts, 0.01, 10.0);

    FILE *f=fopen(argv[1], "rb");
    assert(f!=0);

    int nalu_buf_size=1024*1024;
    uint8_t *nalu_buf=(uint8_t *)malloc(nalu_buf_size);
    uint64_t first_timestamp=0;
    bool first_nalu=true;
    double actual_start_time=profile_time();
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

        while(profile_time()-actual_start_time<time) usleep(1000); // try to play in roughly realtime

        nalu_buf[0]=(len>>24)&0xff;
        nalu_buf[1]=(len>>16)&0xff;
        nalu_buf[2]=(len>>8)&0xff;
        nalu_buf[3]=(len>>0)&0xff;
        track_stream_add_nalus(s.ts, ts/90000.0, nalu_buf, len+4, false);
    }

    track_stream_sync(s.ts);

    const char *stream_stats=track_stream_get_stats(s.ts);
    const char *shared_state_stats=track_shared_state_get_stats(s.tss);
    printf("======== SHARED TRACK STATS ===========\n");
    printf("%s\n\n",shared_state_stats);
    free((void*)shared_state_stats);
    printf("======== STREAM STATS ===========\n");
    printf("%s\n\n", stream_stats);
    free((void*)stream_stats);

    return 0;
}
