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
#include "yaml_stuff.h"
#include "default_setup.h"

typedef struct state {
    track_stream_t *ts;
    uint64_t tracked_frames;
    uint64_t tracked_frames_nonskip;
    uint64_t decoded_macroblocks;
    uint64_t decoded_frames;
    uint64_t face_embeddings;
    double time;
} state_t;

typedef struct {
    const char *video_file_filename;
    int thread_id;
    float track_framerate;
    unsigned int *total_tracked;
    unsigned int *total_tracked_nonskip;
    uint64_t *total_decoded_macroblocks;
    uint64_t *total_face_embeddings;
    double *total_time;
    pthread_mutex_t *lock;
    double duration_sec;
    track_shared_state_t *tss;
} thread_args_t;

static volatile int keep_running = 1;

static void track_result(void *context, track_results_t *r) {
    state_t *s = (state_t *)context;
    if (keep_running)
    {
        if (r->result_type!=TRACK_FRAME_SKIP_FRAMERATE) s->tracked_frames++;
        if (r->result_type == TRACK_FRAME_TRACKED_ROI || r->result_type == TRACK_FRAME_TRACKED_FULL_REFRESH) {
            s->tracked_frames_nonskip++;
        }
        s->time=r->time;
        detection_list_t *track_dets=r->track_dets;
        if (track_dets)
        {
            for(int i=0;i<track_dets->num_detections;i++)
            {
                detection_t *det=track_dets->det[i];
                if (det->face_embedding)
                {
                    if (embedding_get_time(det->face_embedding)==track_dets->time) s->face_embeddings++;
                }
            }
        }
    }
}

static const char *clips[]={
    "/mldata/video/test/clip1_1280x720_5.00fps.hevc",
    "/mldata/video/test/clip2_1280x720_5.00fps.hevc",
    "/mldata/video/test/clip3_1280x720_5.00fps.hevc",
    "/mldata/video/test/clip1_1280x720_5.00fps.264",
    "/mldata/video/test/clip2_1280x720_5.00fps.264",
    "/mldata/video/test/clip3_1280x720_5.00fps.264",
    "/mldata/video/test/clip2_1280x720_5.00fps.264",
    "/mldata/video/test/london_bus_1280x720_5.00fps.hevc",
    //"/mldata/video/test/MOT20-05_1280x832_5.00fps.hevc",
    //"/mldata/video/test/pedestrians_1280x720_5.00fps.hevc",
    //"/mldata/video/test/MOT20-05_1654x1080_25fps.265"
};

static void *run_track_worker(void *arg) {
    cuda_thread_init();
    thread_args_t *args = (thread_args_t *)arg;
    state_t s;
    memset(&s, 0, sizeof(state_t));

    s.ts = track_stream_create(args->tss, &s, track_result);
    track_stream_set_minimum_frame_intervals(s.ts, 1.0/args->track_framerate, 10.0);
    char name[128];
    while(keep_running)
    {
        int num_clips=sizeof(clips)/sizeof(const char *);
        int clip=(rand()&511) % num_clips;
        snprintf(name, sizeof(name), "%s", clips[clip]);
        track_stream_set_name(s.ts, name);
        track_stream_run_video_file(s.ts, clips[clip], SIMPLE_DECODER_CODEC_UNKNOWN, 0.0f, false);
        std::vector<track_results_t *> results=track_stream_get_results(s.ts, true);
        for (auto& res : results) {
            track_results_destroy(res);
        }
    }

    pthread_mutex_lock(args->lock);
    *args->total_tracked_nonskip += s.tracked_frames_nonskip;
    *args->total_tracked += s.tracked_frames;
    *args->total_decoded_macroblocks += s.decoded_macroblocks;
    *args->total_face_embeddings += s.face_embeddings;
    *args->total_time+=s.time;
    pthread_mutex_unlock(args->lock);

    //printf("%s\n",track_stream_get_stats(s.ts));

    track_stream_destroy(s.ts);
    return NULL;
}

typedef struct test_config
{
    const char *testset;
    char name[128];
    const char *yaml_config;
    const char *config_override;
    double duration_sec;
    int cycle;
    int num_threads;
    int infer_w, infer_h;
    float track_framerate;
    //
    float fps;
    float fps_nonskip;
    float mbps;
    float feps;
    float total_time;
} test_config_t;

static void run_stress(test_config_t *config)
{
    track_shared_state_t *shared_state = track_shared_state_create(config->yaml_config);

    infer_config_t inf_config={0};
    inf_config.limit_max_width = config->infer_w;
    inf_config.set_limit_max_width = true;
    inf_config.limit_max_height = config->infer_h;
    inf_config.set_limit_max_height = true;
    track_shared_state_configure_inference(shared_state, &inf_config);

    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * config->num_threads);
    thread_args_t *args = (thread_args_t*)malloc(sizeof(thread_args_t) * config->num_threads);
    keep_running = 1;
    double start_time = profile_time();
    unsigned int total_tracked = 0;
    unsigned int total_tracked_nonskip = 0;
    uint64_t total_decoded_macroblocks = 0;
    uint64_t total_face_embeddings = 0;
    double total_time=0;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

    int num_clips=sizeof(clips)/sizeof(const char *);
    int clip=(rand()&511) % num_clips;

    for (int i = 0; i < config->num_threads; ++i) {
        args[i].video_file_filename = clips[clip];
        args[i].track_framerate = config->track_framerate;
        args[i].thread_id = i;
        args[i].total_tracked = &total_tracked;
        args[i].total_tracked_nonskip = &total_tracked_nonskip;
        args[i].total_decoded_macroblocks = &total_decoded_macroblocks;
        args[i].total_time = &total_time;
        args[i].total_face_embeddings=&total_face_embeddings;
        args[i].lock = &lock;
        args[i].duration_sec = config->duration_sec;
        args[i].tss = shared_state;
        pthread_create(&threads[i], NULL, run_track_worker, &args[i]);
    }

    log_info("Stress cycle %03d running for %f seconds",config->cycle,config->duration_sec);
    sleep((unsigned int)config->duration_sec);
    log_info("Stress cycle %03d finishing....",config->cycle);
    keep_running = 0;
    double elapsed = profile_time()-start_time;

    for (int i = 0; i < config->num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    double avg_fps = total_tracked / config->duration_sec;
    double avg_fps_nonskipped = total_tracked_nonskip / config->duration_sec;
    config->fps=avg_fps;
    config->fps_nonskip=avg_fps_nonskipped;
    config->mbps=((double)total_decoded_macroblocks)/config->duration_sec;
    config->feps=((double)total_face_embeddings)/config->duration_sec;
    config->total_time=total_time/elapsed;
    //printf("MACROS %f SEC %d / %f rate %f 720; %f\n", (double)total_decoded_macroblocks, config->duration_sec, elapsed, config->mbps, config->mbps/3600.0);
    //printf("%40s: %.2f (total) %.2f (nonskip)\n", config->filename, avg_fps, avg_fps_nonskipped);

    track_shared_state_destroy(shared_state);
    free(threads);
    free(args);
}

static const char* get_last_path_part(const char* path) {
    const char* last_slash = strrchr(path, '/');
    if (last_slash) {
        return last_slash + 1;
    } else {
        return path;  // No slash found; return the whole string
    }
}

int main(int argc, char *argv[]) {

    log_debug("ubon_cstuff version = %s", ubon_cstuff_get_version());
    init_cuda_stuff();
    log_debug("Init cuda done");
    log_debug("Initial GPU mem %f",get_process_gpu_mem(false, false));
    image_init();

    std::ostringstream oss, hdr;

    log_set_level(LOG_INFO);

    test_config_t dconfig={0};
    sprintf(dconfig.name, "default");
    dconfig.yaml_config = DEFAULT_TRACKER_YAML;
    dconfig.duration_sec = platform_is_jetson() ? 20 : 30;
    dconfig.track_framerate = 8;
    dconfig.num_threads = platform_is_jetson() ? 8 : 32;
    dconfig.infer_w = 640;
    dconfig.infer_h = 640;
    int cycle=0;
    while(1)
    {
        test_config_t *this_config=&dconfig;
        this_config->cycle=cycle;
        run_stress(this_config);

        auto format_mb = [](double bytes) -> double {
            return bytes / 1e6;  // Convert bytes to MB
        };

        double app_curr = get_cuda_mem(false, false, false);
        double app_hwm = get_cuda_mem(false, true, true);
        cuda_flush();
        double gpu_mem_curr = get_process_gpu_mem(false, false);
        double gpu_mem_hwm = get_process_gpu_mem(true, true);
        int skip_percent = (int)(100 * (1.0 - this_config->fps_nonskip / (this_config->fps + 0.001)));

        oss     << std::setw(44) << this_config->name
                << " " << std::setw(4)  << this_config->num_threads
                << " " << std::setw(8)  << ((int)(this_config->mbps/3600.0))
                << " " << std::setw(5)  << ((int)this_config->fps)
                << " " << std::setw(7)  << std::fixed << std::setprecision(1) << (this_config->total_time)
                << " " << std::setw(5)  << std::fixed << std::setprecision(1) << this_config->feps
                << " " << std::setw(4)  << skip_percent
                << " " << std::setw(8)  << std::fixed << std::setprecision(1) << format_mb(allocation_tracker_get_mem_HWM("image device alloc"))
                << " " << std::setw(8)  << format_mb(allocation_tracker_get_mem_HWM("trt alloc"))
                << " " << std::setw(8)  << format_mb(app_hwm)
                << " " << std::setw(8)  << format_mb(app_curr)
                << " " << std::setw(8)  << format_mb(gpu_mem_hwm)
                << " " << std::setw(8)  << format_mb(gpu_mem_curr)
                << "\n";

        std::cout << oss.str();

        char *s = allocation_tracker_stats();
        allocation_tracker_reset();
        printf("%s\n", s);
        free(s);
        cycle++;
    }
}