
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
#include "trackset.h"
#include "file_decoder.h"

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
        for(int i=0;i<r->track_dets->num_detections;i++)
        {
            detection_t *det=r->track_dets->det[i];
            /*if (det->face_jpeg)
            {
                size_t data_size=0;
                uint8_t *data=jpeg_get_data(det->face_jpeg, &data_size);
                static int n=0;
                char temp[200];
                snprintf(temp, 199, "jpeg_face_%0d.jpg",n++);
                FILE *f=fopen(temp, "wb");
                fwrite(data, 1, data_size, f);
                fclose(f);
                printf("Written face %s; %d bytes\n",temp,(int)data_size);
            }*/
        }

        image_t *img=image_reference(s->img);
        if (img!=0)
        {
            image_t *out_frame_rgb=detection_list_draw(r->track_dets, img);
            printf("%f %f\n",s->img->meta.time, r->time);
            if (out_frame_rgb!=0)
            {
                display_image("video", out_frame_rgb);
                image_destroy(out_frame_rgb);
            }
            image_destroy(img);
        }
        static bool first=true;
        static double first_result_time;
        if (first)
        {
            first=false;
            first_result_time=r->time;
        }
        double target_time=r->time-first_result_time;
        if (realtime)
        {
            while(profile_time()-s->start_time<target_time)
            {
                usleep(1000);
            }
        }
    }
    if (r->track_dets && r->track_dets->frame_jpeg)
    {
        size_t data_size=0;
        uint8_t *data=jpeg_get_data(r->track_dets->frame_jpeg, &data_size);
        static int n=0;
        char temp[200];
        snprintf(temp, 199, "jpeg_%0d.jpg",n++);
        FILE *f=fopen(temp, "wb");
        fwrite(data, 1, data_size, f);
        fclose(f);
        printf("==Written %s; %d bytes\n",temp,(int)data_size);
    }
}

static void process_image(void *context, image_t *img)
{
    state_t *s=(state_t *)context;
    image_t *old_img=s->img;
    s->img=image_reference(img);
    image_destroy(old_img);
    track_stream_run_frame_time(s->ts, img);
    //track_stream_run_single_frame(s->ts, img);
}

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();

    //trackset_t *ts=trackset_load("/mldata/tracking/cevo/annotation/UKof_HD_Indoor_Light_OHcam_001.json");

    state_t s;
    memset(&s, 0, sizeof(state_t));
    s.tss=track_shared_state_create("/mldata/config/track/trackers/uc_reid.yaml");
    s.ts=track_stream_create(s.tss, &s, track_result);
    s.start_time=profile_time();
    track_stream_set_minimum_frame_intervals(s.ts, 0.01, 10.0);

    //decode_file("/mldata/video/test/ind_off_1280x720_7.5fps.264", &s, process_image, 0);
    //track_stream_run_video_file(s.ts, "/mldata/video/test/ind_off_1280x720_7.5fps.264", SIMPLE_DECODER_CODEC_H264, 7.5f, 0);
    //while(1) usleep(1000);

    if (argc>1)
    {
        decode_file(argv[1], &s, process_image, 0);
    }
    else
    {
        webcam_t *w=webcam_create("/dev/video0", 1280, 720);

        while(1)
        {
            image_t *i=webcam_capture(w);
            process_image(&s, i);
            image_destroy(i);
        }
    }
    track_stream_sync(s.ts);
    printf("Done!\n");
    usleep(100000);

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
