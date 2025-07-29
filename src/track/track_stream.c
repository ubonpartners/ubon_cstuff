#include <stdint.h>
#include <stdlib.h>
#include <cassert>
#include <string.h>
#include <unistd.h>
#include <mutex>
#include "track.h"
#include "image.h"
#include "motion_track.h"
#include "log.h"
#include "infer_thread.h"
#include "ctpl_stl.h"
#include "cuda_stuff.h"
#include "BYTETracker.h"
#include "utrack.h"
#include "simple_decoder.h"
#include "yaml_stuff.h"
#include "jpeg_thread.h"
#include "memory_stuff.h"
#include "track_shared.h"
#include "track_aux.h"
#include "rtp_receiver.h"
#include "h26x_assembler.h"
#include "jpeg.h"
#define debugf if (0) log_debug
#define input_debugf if (0) log_error

static std::once_flag initFlag;
static block_allocator_t *track_results_allocator;
static block_allocator_t *ts_queued_job_allocator;

typedef enum ts_queued_job_type
{
    TRACK_STREAM_JOB_IMAGE=0,         // decode image waiting for the main pipeline
    TRACK_STREAM_JOB_VIDEO_DATA=1,    // compressed video data waiting to be decoded (makes possibly multiple TRACK_STREAM_JOB_IMAGE)
    TRACK_STREAM_JOB_JPEG=2,          // compressed JPEG image waiting to be decoded (makes TRACK_STREAM_JOB_IMAGE)
    TRACK_STREAM_JOB_ENCODED_FRAME=3, // complete frames of H26x data waiting to be decoded
    TRACK_STREAM_JOB_SDP=4,           // SDP config
    TRACK_STREAM_JOB_RTP=5,           // RTP packets waiting to be assembled to H26x frames
    TRACK_STREAM_NUM_JOB_TYPES=6
} ts_queued_job_type_t;

//
// [Compressed JPEG]       [RTP Packets]     [SDP]   [Video File]   [Raw images]
//        |                     |             |          |               |
//        |                     v             |          |               |
//        |                 [RTP:Queue5]  [SDP:Queue4]   |               |
//        |                     |             |          |               |
//        v                     v             |          v               |
//  [jpeg:Queue 2]        RTP_RECEIVER<-------+     [File chunker:       |
//        |                     |                    Queue 1]            |
//        |                     v                        |               |
//        |               H26X_ASSEMBLER                 |               |
//        |                     |                        |               |
//        v                     v                        |               |
//   JPEG DECODER     [H26x Frames, Queue 3]             |               |
//        |                     |                        |               |
//        |                     v                        |               |
//        |               SIMPLE_DECODER <---------------+               |
//        |                     |                                        |
//        |            downscale + FR frame skip <-----------------------+
//        |                     |
//        |                     V
//        +----------->[Video frames, Queue 0]
//                     load dependent Rate-limiting
//                              |
//                              V
//  (other streams)       MOTION_TRACKER
//          |             (skip, ROI gen)
//          |                   |
//          |   +---------------+
//          v   v               |                   (other streams)
//      MAIN INFERENCE          |                          |
//              |               v                          V
//              +----------->TRACKER------------------>AUX_INFERENCE
//                              |              (face/clip/FIQA embedding gen
//                              |                  person/face/main jpeg gen)
//                              v                          |
//                    [track results callback] <-----------+

typedef struct ts_queued_job ts_queued_job_t;

struct ts_queued_job
{
    ts_queued_job_t *next;
    ts_queued_job_type_t type;
    double time;
    image_t *img;
    bool single_frame;
    bool is_iframe;
    uint8_t *data;
    size_t data_offset;
    size_t data_len;
    simple_decoder_codec_t codec;
    float fps;
    float start_time, end_time;
};

struct track_stream
{
    track_shared_state_t *tss;
    track_aux_t *taux;
    const char *config_yaml;
    //
    void *result_callback_context;
    void (*result_callback)(void *context, track_results_t *results);
    //
    pthread_mutex_t main_job_mutex;
    uint32_t frame_count;
    bool single_frame; // treat frame as an individual frame, not a video frame
    bool destroying;
    motion_track_t *mt;

    BYTETracker *bytetracker;
    utrack_t *utrack;
    image_format_t stream_image_format;
    //
    double start_time; // ignore frames until >= this time
    double last_run_time;
    double min_time_delta_process;
    double min_time_delta_full_roi;
    bool last_skip;
    roi_t motion_roi;
    roi_t inference_roi;
    roi_t tracked_object_roi;
    image_t *inference_image;
    detection_list_t *inference_detections;
    uint64_t decoded_frames;
    //
    std::vector<track_results_t *> track_results_vec;

    //
    rtp_receiver_t *rtp_receiver;
    h26x_assembler_t *h26x_assembler;
    uint64_t try_schedule_push_count, try_schedule_complete_count;
    pthread_mutex_t input_job_mutex;
    simple_decoder_t *decoder;
    pthread_mutex_t q_mutex;
    int queued_images;
    ts_queued_job_t *jobs[TRACK_STREAM_NUM_JOB_TYPES];
};

static void track_stream_try_process_jobs(track_stream_t *ts);
static void schedule_job_run(track_stream_t *ts);

static void track_stream_init()
{
    track_results_allocator=block_allocator_create("track_result", sizeof(track_results_t));
    ts_queued_job_allocator=block_allocator_create("track_stream_job", sizeof(ts_queued_job_t));
}

track_results_t *track_results_create()
{
    track_results_t *tr=(track_results_t *)block_alloc(track_results_allocator, sizeof(track_results_t));
    memset(tr, 0, sizeof(track_results_t));
    return tr;
}

void track_results_destroy(track_results_t *r)
{
    if (block_reference_count(r)==1)
    {
        if (r->inference_dets) detection_list_destroy(r->inference_dets);
        if (r->track_dets) detection_list_destroy(r->track_dets);
    }
    block_free(r);
}

track_stream_t *track_stream_create(track_shared_state_t *tss,
                                    void *result_callback_context, void (*result_callback)(void *context, track_results_t *results),
                                    const char *config_yaml)
{
    std::call_once(initFlag, track_stream_init);

    track_stream_t *ts=(track_stream_t *)malloc(sizeof(track_stream_t));
    assert(ts!=0);
    memset(ts, 0, sizeof(track_stream_t));

    YAML::Node yaml_base=yaml_merge(tss->config_yaml, config_yaml);
    ts->config_yaml=yaml_to_cstring(yaml_base);

    pthread_mutex_init(&ts->main_job_mutex, 0);
    pthread_mutex_init(&ts->input_job_mutex, 0);
    pthread_mutex_init(&ts->q_mutex, 0);
    ts->frame_count=0;
    ts->tss=tss;
    ts->result_callback_context=result_callback_context;
    ts->result_callback=result_callback;
    ts->mt=motion_track_create(ts->config_yaml);

    std::string tracker_type=yaml_base["tracker_type"].as<std::string>();
    bool use_bytetracker=(strcmp(tracker_type.c_str(), "upyc-bytetrack")==0);
    bool use_utrack=(strcmp(tracker_type.c_str(), "upyc-utrack")==0);
    assert(use_bytetracker||use_utrack); // only supported trackers right now
    if (use_bytetracker)
        ts->bytetracker=new BYTETracker(tss->config_yaml);
    else
        ts->utrack=utrack_create(tss->config_yaml);
    ts->min_time_delta_process=0.0;
    ts->min_time_delta_full_roi=120.0;
    ts->last_skip=false;
    ts->tracked_object_roi=ROI_ZERO;
    ts->stream_image_format=IMAGE_FORMAT_YUV420_DEVICE;
    ts->taux=track_aux_create(tss);
    return ts;
}

image_format_t track_stream_get_stream_image_format(track_stream_t *ts)
{
    return ts->stream_image_format;
}

void track_stream_destroy(track_stream_t *ts)
{
    if (!ts) return;

    // destroy is a bit fiddly to avoid race conditions
    // first we are safe to destroy all input work
    // as long as under mutex

    pthread_mutex_lock(&ts->input_job_mutex);
    pthread_mutex_lock(&ts->q_mutex);
    for(int i=0;i<TRACK_STREAM_NUM_JOB_TYPES;i++)
    {
        ts_queued_job_t *j=ts->jobs[i];
        if (!j) continue;
        ts->jobs[i]=j->next;
        if (j->data) free(j->data);
        if (j->img) destroy_image(j->img);
        block_free(j);
    }
    pthread_mutex_unlock(&ts->q_mutex);
    pthread_mutex_unlock(&ts->input_job_mutex);

    // now lets make sure any 'try work' functions have finished
    // no more should be scheduled because of the above
    while(ts->try_schedule_push_count!=ts->try_schedule_complete_count) usleep(500);

    // now we are safe to destroy the main work stuff
    // destroy the decoder first as it can be a bit annoying
    pthread_mutex_lock(&ts->main_job_mutex);
    pthread_mutex_lock(&ts->input_job_mutex);
    ts->destroying=true;
    simple_decoder_t *dec=ts->decoder;
    if (dec)
    {
        ts->decoder=0;
        simple_decoder_destroy(dec);
    }
    pthread_mutex_unlock(&ts->input_job_mutex);
    pthread_mutex_unlock(&ts->main_job_mutex);

    pthread_mutex_lock(&ts->main_job_mutex);
    pthread_mutex_lock(&ts->q_mutex);
    for(int i=0;i<TRACK_STREAM_NUM_JOB_TYPES;i++)
    {
        ts_queued_job_t *j=ts->jobs[i];
        if (!j) continue;
        ts->jobs[i]=j->next;
        if (j->data) free(j->data);
        if (j->img) destroy_image(j->img);
        block_free(j);
    }
    pthread_mutex_unlock(&ts->q_mutex);
    pthread_mutex_unlock(&ts->main_job_mutex);

    // nothing should be running now
    if (ts->rtp_receiver) rtp_receiver_destroy(ts->rtp_receiver);
    if (ts->h26x_assembler) h26x_assembler_destroy(ts->h26x_assembler);
    if (ts->decoder) simple_decoder_destroy(ts->decoder);
    ts->decoder=0;
    motion_track_destroy(ts->mt);
    ts->mt=0;
    if (ts->bytetracker) delete ts->bytetracker;
    if (ts->utrack) utrack_destroy(ts->utrack);
    ts->utrack=0;
    track_aux_destroy(ts->taux);
    pthread_mutex_destroy(&ts->q_mutex);
    pthread_mutex_destroy(&ts->input_job_mutex);
    pthread_mutex_destroy(&ts->main_job_mutex);
    free((void*)ts->config_yaml);
    free(ts);
}

static void process_results(track_stream_t *ts, track_results_t *r)
{
    track_shared_state_t *tss=ts->tss;
    track_aux_run(ts->taux, ts->inference_image, r->track_dets, ts->single_frame);

    if (ts->result_callback)
    {
        ts->result_callback(ts->result_callback_context, r);
        track_results_destroy(r);
    }
    else
        ts->track_results_vec.push_back(r);
}

static void thread_stream_run_process_inference_results(int id, track_stream_t *ts)
{
    track_results_t *r=track_results_create();

    motion_track_set_roi(ts->mt, ts->inference_roi);
    //detection_list_show(ts->inference_detections);

    r->result_type=TRACK_FRAME_TRACKED_ROI;
    r->time=ts->last_run_time;
    if (ts->bytetracker)
    {
        assert(ts->single_frame==false); // fixme if you want bytetracker to support single frame mode
        r->track_dets=ts->bytetracker->update(ts->inference_detections, ts->last_run_time);
    }
    else if (ts->utrack)
        r->track_dets=utrack_run(ts->utrack, ts->inference_detections, ts->last_run_time, ts->single_frame);
    else
    {
        assert(0);
    }
    r->inference_dets=ts->inference_detections;
    r->motion_roi=ts->motion_roi;
    r->inference_roi=ts->inference_roi;

    if (r->track_dets->num_detections!=0)
    {
        float x0=1.0;
        float x1=0.0;
        float y0=1.0;
        float y1=0.0;
        for(int i=0;i<r->track_dets->num_detections;i++)
        {
            x0=std::min(x0, r->track_dets->det[i]->x0);
            y0=std::min(y0, r->track_dets->det[i]->y0);
            x1=std::min(x1, r->track_dets->det[i]->x1);
            y1=std::min(y1, r->track_dets->det[i]->y1);
        }
        ts->tracked_object_roi.box[0]=x0;
        ts->tracked_object_roi.box[1]=y0;
        ts->tracked_object_roi.box[2]=x1;
        ts->tracked_object_roi.box[3]=y1;
    }

    process_results(ts, r);
    destroy_image(ts->inference_image);
    ts->inference_image=0;
    pthread_mutex_unlock(&ts->main_job_mutex);
    schedule_job_run(ts);
}

static void infer_done_callback(void *context, infer_thread_result_data_t *r)
{
    debugf("infer_done_callback");
    track_stream_t *ts=(track_stream_t *)context;
    track_shared_state_t *tss=ts->tss;
    ts->inference_detections=r->dets;
    ts->inference_roi=r->inference_roi;
    tss->thread_pool->push(thread_stream_run_process_inference_results, ts);
}

void track_stream_set_minimum_frame_intervals(track_stream_t *ts, double min_process, double min_full_roi)
{
    pthread_mutex_lock(&ts->main_job_mutex);
    ts->min_time_delta_process=min_process;
    ts->min_time_delta_full_roi=min_full_roi;
    pthread_mutex_unlock(&ts->main_job_mutex);
}

void track_stream_enable_face_embeddings(track_stream_t *ts, bool enabled, float min_quality)
{
    pthread_mutex_lock(&ts->main_job_mutex);
    track_aux_enable_face_embeddings(ts->taux, enabled, min_quality);
    pthread_mutex_unlock(&ts->main_job_mutex);
}

static void thread_stream_run_input_image_job(int id, track_stream_t *ts, image_t *img, double time, bool single_frame)
{
    debugf("thread_stream_run_input_image_job run");
    track_shared_state_t *tss=ts->tss;
    int scale_w, scale_h;
    double time_delta=0;
    if (ts->destroying)
    {
        destroy_image(img);
        return;
    }
    if ((ts->frame_count==0)||(single_frame)) ts->last_run_time=time-10.0;

    if ((time<ts->last_run_time) || (time>ts->last_run_time+10.0))
    {
        log_warn("unexpected time jump %f->%f; resetting time",ts->last_run_time,time);
        ts->last_run_time=time-10.0;
    }
    time_delta=time-ts->last_run_time+1e-7;
    ts->frame_count++;
    ts->single_frame=single_frame;
    //printf("time %f delta %f min %f skip %d\n",time,time_delta,ts->min_time_delta_process,(time_delta<ts->min_time_delta_process));
    if ((time_delta<ts->min_time_delta_process) || (img==0))
    {
        if (img!=0) destroy_image(img);
        track_results_t *r=track_results_create();
        r->result_type=(img==0) ? TRACK_FRAME_SKIP_NO_IMG : TRACK_FRAME_SKIP_FRAMERATE;
        r->time=time;
        process_results(ts, r);
        pthread_mutex_unlock(&ts->main_job_mutex);
        schedule_job_run(ts);
        return;
    }

    determine_scale_size(img->width, img->height,
                         tss->max_width, tss->max_height, &scale_w, &scale_h,
                         10, 8, 8, false);

    image_t *image_scaled=image_scale_convert(img, ts->stream_image_format, scale_w, scale_h);
    image_check(image_scaled);
    image_check(img);
    destroy_image(img);
    assert(0!=pthread_mutex_trylock(&ts->main_job_mutex));
    assert(ts->destroying==false);
    if (single_frame)
        motion_track_reset(ts->mt);
    else
        motion_track_add_frame(ts->mt, image_scaled);
    image_check(image_scaled);

    if (motion_track_scene_change(ts->mt))
    {
        motion_track_reset(ts->mt);
        if (ts->utrack) utrack_reset(ts->utrack);
    }

    roi_t motion_roi=motion_track_get_roi(ts->mt);
    float skip_roi_thr=(ts->last_skip) ? tss->motiontrack_min_roi_after_skip : tss->motiontrack_min_roi_after_nonskip;
    if (roi_area(&motion_roi)<skip_roi_thr)
    {
        assert(ts->single_frame==false); // should never skip single frames
        debugf("skip inference ROI area %f < %f", roi_area(&motion_roi), skip_roi_thr);
        motion_track_set_roi(ts->mt, ROI_ZERO);
        destroy_image(image_scaled);

        track_results_t *r=track_results_create();
        r->result_type=TRACK_FRAME_SKIP_NO_MOTION;
        r->time=time;
        r->motion_roi=motion_roi;
        r->inference_roi=ROI_ZERO;
        r->track_dets=0;
        r->inference_dets=0;
        ts->last_run_time=time;
        ts->last_skip=true;
        process_results(ts, r);
        assert(0!=pthread_mutex_trylock(&ts->main_job_mutex));
        pthread_mutex_unlock(&ts->main_job_mutex);
        schedule_job_run(ts);
        return;
    }
    ts->last_skip=false;
    // now we are definitely doing a full run
    ts->last_run_time=time;
    ts->motion_roi=motion_roi;

    roi_t expanded_roi=motion_roi;
    if (roi_area(&ts->tracked_object_roi)>0) expanded_roi=roi_union(expanded_roi, ts->tracked_object_roi);

    if (ts->utrack)
    {
        expanded_roi=utrack_predict_positions(ts->utrack, time, ts->mt, expanded_roi);
    }
    assert(ts->inference_image==0);
    ts->inference_image=image_scaled;
    infer_thread_infer_async_callback(tss->infer_thread[INFER_THREAD_DETECTION], image_scaled, expanded_roi, infer_done_callback, ts);
    assert(0!=pthread_mutex_trylock(&ts->main_job_mutex));
}

std::vector<track_results_t *> track_stream_get_results(track_stream_t *ts)
{
    int wait=100;
    while(1)
    {
        pthread_mutex_lock(&ts->input_job_mutex);
        pthread_mutex_lock(&ts->q_mutex);
        bool ok=true;
        for(int i=0;i<TRACK_STREAM_NUM_JOB_TYPES;i++) ok&=(ts->jobs[i]==0);
        pthread_mutex_unlock(&ts->q_mutex);
        pthread_mutex_unlock(&ts->input_job_mutex);
        if (ok) break;
        usleep(wait);
        wait=std::min(5000, wait*2);
    }
    pthread_mutex_lock(&ts->main_job_mutex);
    std::vector<track_results_t *> ret=ts->track_results_vec;
    ts->track_results_vec.clear();
    pthread_mutex_unlock(&ts->main_job_mutex);
    return ret;
}

void track_shared_state_configure_inference(track_shared_state_t *tss, infer_config_t *config)
{
    if (!tss || !config) return;
    infer_thread_configure(tss->infer_thread[INFER_THREAD_DETECTION], config);
}

static void track_stream_try_process_jobs(track_stream_t *ts);

static ts_queued_job_t *ts_queued_job_create()
{
    //input_debugf("input job create!");
    ts_queued_job_t *tj=(ts_queued_job_t *)block_alloc(ts_queued_job_allocator, sizeof(ts_queued_job_t));
    memset(tj, 0, sizeof(ts_queued_job_t));
    return tj;
}

static void track_stream_queue_job(track_stream_t *ts, ts_queued_job_t *new_job)
{
    pthread_mutex_lock(&ts->q_mutex);
    ts_queued_job_t **q=&ts->jobs[(int)new_job->type];
    ts_queued_job_t *job=*q;
    input_debugf("Queue input job type %d",new_job->type);
    if (job==0)
        *q=new_job;
    else
    {
        while(job->next!=0) job=job->next;
        job->next=new_job;
    }
    if (new_job->type==TRACK_STREAM_JOB_IMAGE) ts->queued_images++;
    pthread_mutex_unlock(&ts->q_mutex);
    schedule_job_run(ts);
}

static void track_stream_queue_job_head(track_stream_t *ts, ts_queued_job_t *new_job)
{
    pthread_mutex_lock(&ts->q_mutex);
    ts_queued_job_t **q=&ts->jobs[(int)new_job->type];
    ts_queued_job_t *job=*q;
    input_debugf("Queue input job type %d",new_job->type);
    *q=new_job;
    new_job->next=job;
    if (new_job->type==TRACK_STREAM_JOB_IMAGE) ts->queued_images++;
    pthread_mutex_unlock(&ts->q_mutex);
    schedule_job_run(ts);
}

static void decoder_process_image(void *context, image_t *img)
{
    track_stream_t *ts=(track_stream_t *)context;
    if (ts->destroying) return;
    ts_queued_job_t *job=ts_queued_job_create();
    job->type=TRACK_STREAM_JOB_IMAGE;
    job->img=image_reference(img);
    job->time=img->time;
    job->single_frame=false;
    ts->decoded_frames++;
    input_debugf("got decoded image %dx%d time %f\n",img->width,img->height,img->time);
    track_stream_queue_job(ts, job);
}

bool track_stream_run_on_jpeg(track_stream_t *ts, uint8_t *jpeg_data, int jpeg_data_length)
{
    ts_queued_job_t *job=ts_queued_job_create();
    job->type=TRACK_STREAM_JOB_JPEG;
    job->data=(uint8_t *)malloc(jpeg_data_length);
    job->data_len=jpeg_data_length;
    memcpy(job->data, jpeg_data, jpeg_data_length);
    track_stream_queue_job(ts, job);
    return true;
}

static void rtp_packet_callback(void *context, const rtp_packet_t *pkt)
{
    // an in-order/decrypted rtp packet received from the rtp receiver
    // we pass it on to be assembled into a video frame
    track_stream_t *ts=(track_stream_t *)context;
    if (ts->h26x_assembler) h26x_assembler_process_rtp(ts->h26x_assembler, pkt);
}

static void h26x_frame_callback(void *context, const h26x_frame_descriptor_t *desc)
{
    // a complete frame of NALUs received from the h26x assembler
    // we queue it for decode
    track_stream_t *ts=(track_stream_t *)context;
    ts_queued_job_t *job=ts_queued_job_create();
    job->type=TRACK_STREAM_JOB_ENCODED_FRAME;
    job->data=(uint8_t *)malloc(desc->annexb_length);
    memcpy(job->data, desc->annexb_data, desc->annexb_length);
    job->time=desc->extended_rtp_timestamp/90000.0;
    job->is_iframe=desc->nal_stats.idr_count!=0;
    track_stream_queue_job(ts, job);
}

static void track_stream_try_process_jobs(track_stream_t *ts)
{
    while(1)
    {
        for(int t=0;t<TRACK_STREAM_NUM_JOB_TYPES;t++)
        {
            if (ts->jobs[t]==0) continue;
            input_debugf("Loop type %d",t);
            if (t>0 && ts->queued_images>2) continue;
            pthread_mutex_t *mut=(t==0) ? &ts->main_job_mutex : &ts->input_job_mutex;

            if (0!=pthread_mutex_trylock(mut))
            {
                input_debugf("Mutex is locked");
                continue; // come back later, already busy
            }

            if (ts->destroying)
            {
                pthread_mutex_unlock(mut);
                continue;
            }

            input_debugf("Processing type %d",t);
            pthread_mutex_lock(&ts->q_mutex);
            ts_queued_job_t **q=&ts->jobs[t];
            ts_queued_job_t *job=*q;
            if (job!=0)
                *q=job->next;
            else
                *q=0;
            if (t==TRACK_STREAM_JOB_IMAGE) ts->queued_images--;
            pthread_mutex_unlock(&ts->q_mutex);
            if (job==0)
            {
                //input_debugf("track_stream_try_process_input_jobs - no job");
                pthread_mutex_unlock(mut);
                continue;
            }
            job->next=0;

            switch(job->type)
            {
                case TRACK_STREAM_JOB_IMAGE:
                {
                    track_shared_state_t *tss=ts->tss;
                    image_t *img=job->img;
                    double time=job->time;
                    bool single_frame=job->single_frame;
                    block_free(job);
                    input_debugf("=====> running main decoder job\n");
                    assert(ts->destroying==false);
                    tss->thread_pool->push(thread_stream_run_input_image_job, ts, img, time, single_frame);
                    continue; // we DON'T unlock the mutex here, it's held until inference/tracking complete
                }
                case TRACK_STREAM_JOB_VIDEO_DATA:
                {
                    input_debugf("running input video data job");
                    assert(ts->destroying==false);
                    if (ts->decoder==0)
                    {
                        printf("Create decoder");
                        ts->decoder=simple_decoder_create(ts, decoder_process_image, job->codec);
                        simple_decoder_set_framerate(ts->decoder, job->fps);
                        if (job->end_time!=0) simple_decoder_set_max_time(ts->decoder, job->end_time);
                    }

                    uint64_t decoded_frames=ts->decoded_frames;
                    for (int i=0;i<32;i++)
                    {
                        size_t data_len=std::min((size_t)8192, job->data_len-job->data_offset);
                        //
                        input_debugf("Decode video %d/%d",(int)job->data_offset,(int)job->data_len);
                        simple_decoder_decode(ts->decoder, job->data+job->data_offset, data_len);
                        job->data_offset+=data_len;
                        if (ts->decoded_frames!=decoded_frames) break;
                    }

                    if (job->data_offset<job->data_len)
                    {
                        // requeue remaining video data
                        track_stream_queue_job_head(ts, job);
                    }
                    else
                    {
                        free(job->data);
                        block_free(job);
                    }
                    //input_debugf("running input video data job done");
                    break;
                }
                case TRACK_STREAM_JOB_JPEG:
                {
                    image_t *img=decode_jpeg(job->data, job->data_len);
                    free(job->data);
                    if (!img)
                    {
                        img=create_image(128, 128, IMAGE_FORMAT_YUV420_DEVICE);
                        // fixme - we need to always give a results callback for jpeg even if corrupt/missing/...
                        // better to explicitly flag this
                    }
                    // queue decoded image for further processing
                    ts_queued_job_t *out_job=ts_queued_job_create();
                    out_job->type=TRACK_STREAM_JOB_IMAGE;
                    out_job->img=img;
                    out_job->time=job->time;
                    out_job->single_frame=true;
                    block_free(job);
                    track_stream_queue_job(ts, out_job);
                    break;
                }
                case TRACK_STREAM_JOB_ENCODED_FRAME:
                {
                    if (ts->decoder) simple_decoder_decode(ts->decoder, job->data, job->data_len);
                    free(job->data);
                    block_free(job);
                    break;
                }
                case TRACK_STREAM_JOB_SDP:
                {
                    if (!ts->rtp_receiver) ts->rtp_receiver=rtp_receiver_create(ts, rtp_packet_callback);
                    set_sdp_t sdp;
                    memset(&sdp, 0, sizeof(set_sdp_t));
                    if (0==rtp_receiver_set_sdp(ts->rtp_receiver, (const char *)job->data, &sdp))
                    {
                        if (sdp.is_h264 || sdp.is_h265)
                        {
                            if (ts->h26x_assembler)  h26x_assembler_destroy(ts->h26x_assembler);
                            ts->h26x_assembler=h26x_assembler_create(sdp.is_h264 ? H26X_CODEC_H264 : H26X_CODEC_H265,
                                                                     ts, h26x_frame_callback);
                            if (ts->decoder) simple_decoder_destroy(ts->decoder);
                            ts->decoder=simple_decoder_create(ts, decoder_process_image, sdp.is_h264 ? SIMPLE_DECODER_CODEC_H264 : SIMPLE_DECODER_CODEC_H265);
                        }
                        else // TODO: OPUS
                        {
                            log_error("SDP configured is not H264/H265");
                        }
                    }
                    else
                    {
                        log_error("rtp_receiver_set_sdp failed");
                    }
                    free(job->data);
                    block_free(job);
                    break;
                }
                case TRACK_STREAM_JOB_RTP:
                {
                    if (ts->rtp_receiver)
                    {
                        int offs=0;
                        while(offs<job->data_len)
                        {
                            uint32_t len=*((uint32_t *)(job->data+offs));
                            offs+=4;
                            assert(offs+len<=job->data_len);
                            rtp_receiver_add_packet(ts->rtp_receiver, job->data+offs, len);
                        }
                    }
                    free(job->data);
                    block_free(job);
                    break;
                }
                default:
                {
                    break;
                }
            }
            pthread_mutex_unlock(mut);
        } // job type loop

        break;
    }
    (void)__atomic_add_fetch(&ts->try_schedule_complete_count, 1, __ATOMIC_RELAXED);
}

static void track_stream_try_process_jobs_id(int id, track_stream_t *ts)
{
    track_stream_try_process_jobs(ts);
}

static void schedule_job_run(track_stream_t *ts)
{
    track_shared_state_t *tss=ts->tss;
    (void)__atomic_add_fetch(&ts->try_schedule_push_count, 1, __ATOMIC_RELAXED);
    tss->thread_pool->push(track_stream_try_process_jobs_id, ts);
}

static void do_track_stream_run(track_stream_t *ts, image_t *img, double time, bool single_frame)
{
    while(ts->queued_images>2)
    {
        usleep(1000); // backpressure
    }
    ts_queued_job_t *job=ts_queued_job_create();
    job->type=TRACK_STREAM_JOB_IMAGE;
    job->img=image_reference(img);
    job->time=time;
    job->single_frame=single_frame;
    track_stream_queue_job(ts, job);
}

void track_stream_run(track_stream_t *ts, image_t *img, double time)
{
    do_track_stream_run(ts, img, time, false);
}

void track_stream_run_frame_time(track_stream_t *ts, image_t *img)
{
    assert(img!=0);
    do_track_stream_run(ts, img, img->time, false);
}

void track_stream_run_single_frame(track_stream_t *ts, image_t *img)
{
    assert(img!=0);
    do_track_stream_run(ts, img, img->time, true);
}

static void track_run_video_process_image(void *context, image_t *img)
{
    track_stream_t *ts=(track_stream_t *)context;
    if (img!=0 && img->time>=ts->start_time) track_stream_run_frame_time(ts, img);
}

extern float file_decoder_parse_fps(const char *s);
extern simple_decoder_codec_t file_decoder_parse_codec(const char *file);

void track_stream_run_video_file(track_stream_t *ts, const char *file, simple_decoder_codec_t codec, double video_fps, double start_time, double end_time)
{
    if (codec==SIMPLE_DECODER_CODEC_UNKNOWN)
    {
        codec=file_decoder_parse_codec(file);
        if (codec==SIMPLE_DECODER_CODEC_UNKNOWN)
        {
            log_error("track_stream_run_video_file : no codec");
            return;
        }
    }
    if (video_fps==0.0f)
    {
        video_fps=file_decoder_parse_fps(file);
        if (video_fps==0.0f)
        {
            log_error("track_stream_run_video_file : no fps");
            return;
        }
    }
    input_debugf("track_stream_run_video_file %s",file);
    FILE *fp = fopen(file, "rb");
    if (!fp)
    {
        log_error("Failed to open input file %s", file);
        return;
    }

    fseek(fp, 0L, SEEK_END);
    size_t sz = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    void *mem=malloc(sz);
    if (!mem)
    {
        log_error("Failed to allocate %ld bytes for input file %s", sz, file);
        fclose(fp);
        return;
    }
    bool ok=sz==fread(mem, 1, sz, fp);
    fclose(fp);
    if (!ok)
    {
        log_error("Failed to read input file %s", file);
        free(mem);
        return;
    }
    ts_queued_job_t *job=ts_queued_job_create();
    job->type=TRACK_STREAM_JOB_VIDEO_DATA;
    job->data=(uint8_t *)mem;
    job->data_offset=0;
    job->data_len=sz;
    job->codec=codec;
    job->fps=video_fps;
    job->start_time=start_time;
    job->end_time=end_time;
    track_stream_queue_job(ts, job);
}

void track_stream_set_sdp(track_stream_t *ts, const char *sdp_str)
{
    int len=strlen(sdp_str)+1;
    uint8_t *mem=(uint8_t *)malloc(len+4);
    if (mem==0) return;
    *((uint32_t *)(mem))=len;
    memcpy(mem+4, sdp_str, len);
    ts_queued_job_t *job=ts_queued_job_create();
    job->type=TRACK_STREAM_JOB_SDP;
    job->data=(uint8_t *)mem;
    job->data_offset=0;
    job->data_len=len;
    track_stream_queue_job(ts, job);
}

void track_stream_add_rtp_packets(track_stream_t *ts, int num_packets, uint8_t **data, int *length)
{
    int tlen=0;
    for(int i=0;i<num_packets;i++) tlen+=(4+length[i]);
    uint8_t *mem=(uint8_t *)malloc(tlen);
    if (mem==0) return;
    int offs=0;
    for(int i=0;i<num_packets;i++)
    {
        *((uint32_t *)(mem+offs))=length[i];
        offs+=4;
        memcpy(mem+offs, data[i], length[i]);
        offs+=length[i];
    }
    assert(offs==tlen);
    ts_queued_job_t *job=ts_queued_job_create();
    job->type=TRACK_STREAM_JOB_RTP;
    job->data=(uint8_t *)mem;
    job->data_offset=0;
    job->data_len=tlen;
    track_stream_queue_job(ts, job);
}