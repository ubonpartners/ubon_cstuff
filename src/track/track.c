#include <stdint.h>
#include <stdlib.h>
#include <cassert>
#include <string.h>
#include <unistd.h>
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

#define debugf if (0) log_debug

struct track_shared_state
{
    const char *config_yaml;
    int max_width, max_height;
    int num_worker_threads;
    float motiontrack_min_roi;
    infer_thread_t *infer_thread;
    const char *tracker_type;
    model_description_t *md;
    ctpl::thread_pool *thread_pool;
};

struct track_stream
{
    track_shared_state_t *tss;
    //
    void *result_callback_context;
    void (*result_callback)(void *context, track_results_t *results);
    //
    pthread_mutex_t run_mutex;
    uint32_t frame_count;
    motion_track_t *mt;

    BYTETracker *bytetracker;
    utrack_t *utrack;
    image_format_t stream_image_format;
    //
    double start_time; // ignore frames until >= this time
    double last_run_time;
    double min_time_delta_process;
    double min_time_delta_full_roi;
    roi_t motion_roi;
    roi_t inference_roi;
    roi_t tracked_object_roi;
    detection_list_t *inference_detections;
    //
    std::vector<track_results_t> track_results;
};

static void ctpl_thread_init(int id, pthread_barrier_t *barrier)
{
    debugf("starting ctpl thread %d",id);
    cuda_thread_init();
    pthread_barrier_wait(barrier);
}

track_shared_state_t *track_shared_state_create(const char *yaml_config)
{
    YAML::Node yaml_base=yaml_load(yaml_config);
    std::string tracker_type=yaml_base["tracker_type"].as<std::string>();

    track_shared_state_t *tss=(track_shared_state_t *)malloc(sizeof(track_shared_state_t));
    assert(tss!=0);
    memset(tss, 0, sizeof(track_shared_state_t));
    debugf("Track shared state create");
    tss->config_yaml=yaml_to_cstring(yaml_base);
    tss->max_width=yaml_get_int_value(yaml_base["max_width"], 1280);
    tss->max_height=yaml_get_int_value(yaml_base["max_height"], 1280);
    tss->num_worker_threads=yaml_get_int_value(yaml_base["num_worker_threads"], 4);
    tss->motiontrack_min_roi=yaml_get_float_value(yaml_base["motiontrack_min_roi"], 0.05);
    tss->tracker_type=strdup(tracker_type.c_str());
    // create worker threads
    tss->thread_pool=new ctpl::thread_pool(tss->num_worker_threads);
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, tss->num_worker_threads);
    for(int i=0;i<tss->num_worker_threads;i++) tss->thread_pool->push(ctpl_thread_init, &barrier);

    // setup inference from yaml config
    YAML::Node inferenceConfigNode = yaml_base["inference_config"];
    std::string trt_file=inferenceConfigNode["trt"].as<std::string>();
    const char *inference_yaml=yaml_to_cstring(inferenceConfigNode);
    infer_config_t config={};
    config.det_thr=yaml_get_float_value(yaml_base["conf_thr"], 0.05);
    config.set_det_thr=true;
    config.nms_thr=yaml_get_float_value(yaml_base["nms_thr"], 0.45);
    config.set_nms_thr=true;
    tss->infer_thread=infer_thread_start(trt_file.c_str(), inference_yaml, &config);
    tss->md=infer_thread_get_model_description(tss->infer_thread);
    free((void*)inference_yaml);

    // done
    return tss;
}

model_description_t *track_shared_state_get_model_description(track_shared_state_t *tss)
{
    return tss->md;
}

void track_shared_state_destroy(track_shared_state_t *tss)
{
    if (!tss) return;
    tss->thread_pool->stop();
    delete tss->thread_pool;
    infer_thread_destroy(tss->infer_thread);
    free((void *)tss->config_yaml);
    free((void *)tss->tracker_type);
    free(tss);
}

track_stream_t *track_stream_create(track_shared_state_t *tss, void *result_callback_context, void (*result_callback)(void *context, track_results_t *results))
{
    track_stream_t *ts=(track_stream_t *)malloc(sizeof(track_stream_t));
    assert(ts!=0);
    memset(ts, 0, sizeof(track_stream_t));
    pthread_mutex_init(&ts->run_mutex, 0);
    ts->frame_count=0;
    ts->tss=tss;
    ts->result_callback_context=result_callback_context;
    ts->result_callback=result_callback;
    ts->mt=motion_track_create(tss->config_yaml);
    bool use_bytetracker=(strcmp(tss->tracker_type, "upyc-bytetrack")==0);
    bool use_utrack=(strcmp(tss->tracker_type, "upyc-utrack")==0);
    assert(use_bytetracker||use_utrack); // only supported trackers right now
    if (use_bytetracker)
        ts->bytetracker=new BYTETracker(tss->config_yaml);
    else
        ts->utrack=utrack_create(tss->config_yaml);
    ts->min_time_delta_process=0.0;
    ts->min_time_delta_full_roi=120.0;
    ts->tracked_object_roi=ROI_ZERO;
    ts->stream_image_format=IMAGE_FORMAT_YUV420_DEVICE;
    return ts;
}

image_format_t track_stream_get_stream_image_format(track_stream_t *ts)
{
    return ts->stream_image_format;
}

void track_stream_destroy(track_stream_t *ts)
{
    if (!ts) return;
    pthread_mutex_lock(&ts->run_mutex);
    pthread_mutex_unlock(&ts->run_mutex);
    motion_track_destroy(ts->mt);
    if (ts->bytetracker) delete ts->bytetracker;
    if (ts->utrack) utrack_destroy(ts->utrack);
    pthread_mutex_destroy(&ts->run_mutex);
    free(ts);
}

static void process_results(track_stream_t *ts, track_results_t *r)
{
    if (ts->result_callback)
        ts->result_callback(ts->result_callback_context, r);
    else
        ts->track_results.push_back(*r);
}

static void thread_stream_run_process_inference_results(int id, track_stream_t *ts)
{
    track_results_t r;
    memset(&r, 0, sizeof(track_results_t));

    motion_track_set_roi(ts->mt, ts->inference_roi);
    //detection_list_show(ts->inference_detections);

    r.result_type=TRACK_FRAME_TRACKED_ROI;
    r.time=ts->last_run_time;
    if (ts->bytetracker)
        r.track_dets=ts->bytetracker->update(ts->inference_detections, ts->last_run_time);
    else if (ts->utrack)
        r.track_dets=utrack_run(ts->utrack, ts->inference_detections, ts->last_run_time);
    else
    {
        assert(0);
    }
    r.inference_dets=ts->inference_detections;
    r.motion_roi=ts->motion_roi;
    r.inference_roi=ts->inference_roi;

    if (r.track_dets->num_detections!=0)
    {
        float x0=1.0;
        float x1=0.0;
        float y0=1.0;
        float y1=0.0;
        for(int i=0;i<r.track_dets->num_detections;i++)
        {
            x0=std::min(x0, r.track_dets->det[i]->x0);
            y0=std::min(y0, r.track_dets->det[i]->y0);
            x1=std::min(x1, r.track_dets->det[i]->x1);
            y1=std::min(y1, r.track_dets->det[i]->y1);
        }
        ts->tracked_object_roi.box[0]=x0;
        ts->tracked_object_roi.box[1]=y0;
        ts->tracked_object_roi.box[2]=x1;
        ts->tracked_object_roi.box[3]=y1;
    }

    process_results(ts, &r);
    pthread_mutex_unlock(&ts->run_mutex);
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
    pthread_mutex_lock(&ts->run_mutex);
    ts->min_time_delta_process=min_process;
    ts->min_time_delta_full_roi=min_full_roi;
    pthread_mutex_unlock(&ts->run_mutex);
}

static void thread_stream_run_input_job(int id, track_stream_t *ts, image_t *img, double time)
{
    debugf("thread_stream_run_input_job run");
    track_shared_state_t *tss=ts->tss;
    int scale_w, scale_h;
    double time_delta=0;
    if (ts->frame_count==0) ts->last_run_time=time-10.0;

    if ((time<ts->last_run_time) || (time>=ts->last_run_time+5.0))
    {
        log_warn("unexpected time jump %f->%f; resetting time",ts->last_run_time,time);
        ts->last_run_time=time-10.0;
    }
    time_delta=time-ts->last_run_time+1e-7;
    ts->frame_count++;
    //printf("time %f delta %f min %f skip %d\n",time,time_delta,ts->min_time_delta_process,(time_delta<ts->min_time_delta_process));
    if ((time_delta<ts->min_time_delta_process) || (img==0))
    {
        if (img!=0) destroy_image(img);
        track_results_t r;
        memset(&r, 0, sizeof(track_results_t));
        r.result_type=(img==0) ? TRACK_FRAME_SKIP_NO_IMG : TRACK_FRAME_SKIP_FRAMERATE;
        r.time=time;
        r.motion_roi=ROI_ZERO;
        r.inference_roi=ROI_ZERO;
        r.track_dets=0;
        r.inference_dets=0;
        process_results(ts, &r);
        pthread_mutex_unlock(&ts->run_mutex);
        return;
    }

    determine_scale_size(img->width, img->height,
                         tss->max_width, tss->max_height, &scale_w, &scale_h,
                         10, 8, 8, false);

    image_t *image_scaled=image_scale_convert(img, ts->stream_image_format, scale_w, scale_h);
    image_check(img);
    image_check(image_scaled);
    image_check(img);
    destroy_image(img);
    motion_track_add_frame(ts->mt, image_scaled);
    image_check(image_scaled);

    roi_t motion_roi=motion_track_get_roi(ts->mt);
    if (roi_area(&motion_roi)<tss->motiontrack_min_roi)
    {
        debugf("skip inference ROI area %f < %f", roi_area(&motion_roi), tss->motiontrack_min_roi);
        motion_track_set_roi(ts->mt, ROI_ZERO);
        destroy_image(image_scaled);

        track_results_t r;
        memset(&r, 0, sizeof(track_results_t));
        r.result_type=TRACK_FRAME_SKIP_NO_MOTION;
        r.time=time;
        r.motion_roi=motion_roi;
        r.inference_roi=ROI_ZERO;
        r.track_dets=0;
        r.inference_dets=0;
        ts->last_run_time=time;

        process_results(ts, &r);
        pthread_mutex_unlock(&ts->run_mutex);
        return;
    }

    // now we are definitely doing a full run
    ts->last_run_time=time;
    ts->motion_roi=motion_roi;

    roi_t expanded_roi=motion_roi;
    if (roi_area(&ts->tracked_object_roi)>0) expanded_roi=roi_union(expanded_roi, ts->tracked_object_roi);

    if (ts->utrack)
    {
        expanded_roi=utrack_predict_positions(ts->utrack, time, ts->mt, expanded_roi);
    }

    infer_thread_infer_async_callback(tss->infer_thread, image_scaled, expanded_roi, infer_done_callback, ts);
    destroy_image(image_scaled);
}

void track_stream_run(track_stream_t *ts, image_t *img, double time)
{
    pthread_mutex_lock(&ts->run_mutex);
    debugf("track stream run");
    track_shared_state_t *tss=ts->tss;
    tss->thread_pool->push(thread_stream_run_input_job, ts, image_reference(img), time);
}

void track_stream_run_frame_time(track_stream_t *ts, image_t *img)
{
    assert(img!=0);
    track_stream_run(ts, img, img->time);
}

static void track_run_video_process_image(void *context, image_t *img)
{
    track_stream_t *ts=(track_stream_t *)context;
    if (img!=0 && img->time>=ts->start_time) track_stream_run_frame_time(ts, img);
}

void track_stream_run_video_file(track_stream_t *ts, const char *file, simple_decoder_codec_t codec, double video_fps, double start_time, double end_time)
{
    FILE *input = fopen(file, "rb");
    if (!input)
    {
        log_error("Failed to open input file %s", input);
        return;
    }

    simple_decoder_t *decoder = simple_decoder_create(ts, track_run_video_process_image, codec);
    ts->start_time=start_time;
    simple_decoder_set_framerate(decoder, video_fps);
    simple_decoder_set_max_time(decoder, end_time);
    uint8_t buffer[4096];
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), input)) > 0)
    {
        simple_decoder_decode(decoder, buffer, bytes_read);
    }

    simple_decoder_destroy(decoder);
    fclose(input);
}

std::vector<track_results_t> track_stream_get_results(track_stream_t *ts)
{
    pthread_mutex_lock(&ts->run_mutex);
    std::vector<track_results_t> ret=ts->track_results;
    ts->track_results.clear();
    pthread_mutex_unlock(&ts->run_mutex);
    return ret;
}

void track_shared_state_configure_inference(track_shared_state_t *tss, infer_config_t *config)
{
    if (!tss || !config) return;
    infer_thread_configure(tss->infer_thread, config);
}
