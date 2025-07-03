
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

typedef struct state
{
    track_shared_state_t *tss;
    track_stream_t *ts;
    image_t *img;
    double start_time;
} state_t;

static void track_result(void *context, track_results_t *r)
{
    state_t *s=(state_t *)context;
    printf("result type %d time %f\n",r->result_type,r->time);
    if (r->track_dets!=0)
    {
        //detection_list_show(r->track_dets);
        image_t *img=image_reference(s->img);
        image_t *out_frame_rgb=detection_list_draw(r->track_dets, img);
        display_image("video", out_frame_rgb);
        double target_time=r->time;
        while(profile_time()-s->start_time<target_time)
        {
            usleep(1000);
        }
        destroy_image(out_frame_rgb);
        destroy_image(img);
    }
}

static void process_image(void *context, image_t *img)
{
    state_t *s=(state_t *)context;
    image_t *old_img=s->img;
    s->img=image_reference(img);
    destroy_image(old_img);
    track_stream_run_frame_time(s->ts, img);
}

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();

    trackset_t *ts=trackset_load("/mldata/tracking/cevo/annotation/UKof_HD_Indoor_Light_OHcam_001.json");

    state_t s;
    memset(&s, 0, sizeof(state_t));
    s.tss=track_shared_state_create("/mldata/config/track/trackers/uc_bytetrack.yaml");
    s.ts=track_stream_create(s.tss, &s, track_result);
    s.start_time=profile_time();
    track_stream_set_minimum_frame_intervals(s.ts, 0.01, 10.0);

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
            destroy_image(i);
        }
    }


    return 0;
}
