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

typedef struct state {
    track_stream_t *ts;
    uint32_t tracked_frames;
    uint32_t tracked_frames_nonskip;
} state_t;

typedef struct {
    const char *filename;
    int thread_id;
    unsigned int *total_tracked;
    unsigned int *total_tracked_nonskip;
    pthread_mutex_t *lock;
    double duration_sec;
    track_shared_state_t *tss;
} thread_args_t;

static volatile int keep_running = 1;

static void track_result(void *context, track_results_t *r) {
    state_t *s = (state_t *)context;
    if (r->result_type!=TRACK_FRAME_SKIP_FRAMERATE) s->tracked_frames++;
    if (r->result_type == TRACK_FRAME_TRACKED_ROI || r->result_type == TRACK_FRAME_TRACKED_FULL_REFRESH) {
        s->tracked_frames_nonskip++;
    }
    if (r->track_dets != 0) detection_list_destroy(r->track_dets);
    if (r->inference_dets != 0) detection_list_destroy(r->inference_dets);
}

static void process_image(void *context, image_t *img) {
    state_t *s = (state_t *)context;
    track_stream_run_frame_time(s->ts, img);
}

static void *run_track_worker(void *arg) {
    cuda_thread_init();
    thread_args_t *args = (thread_args_t *)arg;
    state_t s;
    memset(&s, 0, sizeof(state_t));

    s.ts = track_stream_create(args->tss, &s, track_result);
    track_stream_set_minimum_frame_intervals(s.ts, 0.01, 10.0);

    FILE *input = fopen(args->filename, "rb");
    if (!input) {
        log_fatal("Failed to open input file %s", args->filename);
        return NULL;
    }

    simple_decoder_t *decoder = simple_decoder_create(&s, process_image, SIMPLE_DECODER_CODEC_H264);
    simple_decoder_set_output_format(decoder, track_stream_get_stream_image_format(s.ts));
    uint8_t buffer[4096];

    while (keep_running) {
        size_t bytes_read = fread(buffer, 1, sizeof(buffer), input);
        if (bytes_read <= 0) {
            fseek(input, 0, SEEK_SET);
            continue;
        }
        simple_decoder_decode(decoder, buffer, bytes_read);
    }

    pthread_mutex_lock(args->lock);
    *args->total_tracked_nonskip += s.tracked_frames_nonskip;
    *args->total_tracked += s.tracked_frames;
    pthread_mutex_unlock(args->lock);

    simple_decoder_destroy(decoder);
    fclose(input);
    track_stream_destroy(s.ts);
    return NULL;
}

typedef struct test_config
{
    const char *filename;
    int duration_sec;
    int num_threads;
    //
    float fps;
    float fps_nonskip;
} test_config_t;

static void run_one_test(test_config_t *config)
{
    track_shared_state_t *shared_state = track_shared_state_create("/mldata/config/track/trackers/uc_test.yaml");

    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * config->num_threads);
    thread_args_t *args = (thread_args_t*)malloc(sizeof(thread_args_t) * config->num_threads);
    keep_running = 1;
    unsigned int total_tracked = 0;
    unsigned int total_tracked_nonskip = 0;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

    for (int i = 0; i < config->num_threads; ++i) {
        args[i].filename = config->filename;
        args[i].thread_id = i;
        args[i].total_tracked = &total_tracked;
        args[i].total_tracked_nonskip = &total_tracked_nonskip;
        args[i].lock = &lock;
        args[i].duration_sec = config->duration_sec;
        args[i].tss = shared_state;
        pthread_create(&threads[i], NULL, run_track_worker, &args[i]);
    }

    sleep((unsigned int)config->duration_sec);
    keep_running = 0;

    for (int i = 0; i < config->num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    double avg_fps = total_tracked / config->duration_sec;
    double avg_fps_nonskipped = total_tracked_nonskip / config->duration_sec;
    config->fps=avg_fps;
    config->fps_nonskip=avg_fps_nonskipped;
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

    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();

    test_config_t config={0};

    std::ostringstream oss;

    for(int tf=0;tf<2;tf++)
    {
        for(int th=0;th<5;th++)
        {
            config.filename=(tf==0) ? "/mldata/video/bc1.264" : "/mldata/video/MOT20-05.264";
            config.duration_sec=10;
            config.num_threads=1<<th;
            run_one_test(&config);

            auto format_size = [](double bytes) -> std::string {
                double mb = bytes / 1e6;
                char buf[32];
                if (mb >= 1000.0)
                    std::snprintf(buf, sizeof(buf), "%.2fGB", mb / 1000.0);
                else
                    std::snprintf(buf, sizeof(buf), "%.1fMB", mb);
                return std::string(buf);
            };

            double app_curr=get_cuda_mem(false, false, false);
            double app_hwm=get_cuda_mem(false, true, true);

            oss << "Video " << std::setw(40) << get_last_path_part(config.filename)
                << " thr: " << std::setw(4) << config.num_threads
                << " fps: " << std::setw(4) << config.fps
                << " fps (nonskip): " << std::setw(4) << config.fps_nonskip
                << " ImgMem: " << std::setw(8) << format_size(allocation_tracker_get_mem_HWM("image device alloc"))
                << " TRTMem: " << std::setw(8) << format_size(allocation_tracker_get_mem_HWM("trt alloc"))
                << " Cuda HWM: " << std::setw(8) << format_size(app_hwm)
                << " Cuda Curr: " << std::setw(8) << format_size(app_curr)
                << "\n";

            std::cout << oss.str();

            char *s=allocation_tracker_stats();
            allocation_tracker_reset();
            printf("%s\n",s);
            free(s);
        }
    }
}

/*
Video                                  bc1.264 thr:    1 fps:  300 fps (nonskip):  271
Video                                  bc1.264 thr:    2 fps:  321 fps (nonskip):  293
Video                                  bc1.264 thr:    4 fps:  486 fps (nonskip):  437
Video                                  bc1.264 thr:    8 fps:  631 fps (nonskip):  532
Video                             MOT20-05.264 thr:    1 fps:  167 fps (nonskip):  150
Video                             MOT20-05.264 thr:    2 fps:  245 fps (nonskip):  218
Video                             MOT20-05.264 thr:    4 fps:  317 fps (nonskip):  275
Video                             MOT20-05.264 thr:    8 fps:  366 fps (nonskip):  331
==
Video                                  bc1.264 thr:    1 fps:  302 fps (nonskip):  256 ImgMem:    7.8MB TRTMem:  246.6MB
Video                                  bc1.264 thr:    2 fps:  305 fps (nonskip):  279 ImgMem:   12.5MB TRTMem:  246.6MB
Video                                  bc1.264 thr:    4 fps:  424 fps (nonskip):  374 ImgMem:   27.8MB TRTMem:  246.6MB
Video                                  bc1.264 thr:    8 fps:  581 fps (nonskip):  481 ImgMem:   52.6MB TRTMem:  246.6MB
Video                                  bc1.264 thr:   16 fps:  656 fps (nonskip):  460 ImgMem:   99.3MB TRTMem:  246.6MB
Video                             MOT20-05.264 thr:    1 fps:  230 fps (nonskip):  157 ImgMem:    7.6MB TRTMem:  246.6MB
Video                             MOT20-05.264 thr:    2 fps:  324 fps (nonskip):  220 ImgMem:   14.8MB TRTMem:  246.6MB
Video                             MOT20-05.264 thr:    4 fps:  460 fps (nonskip):  316 ImgMem:   29.7MB TRTMem:  246.6MB
Video                             MOT20-05.264 thr:    8 fps:  588 fps (nonskip):  404 ImgMem:   56.9MB TRTMem:  246.6MB
Video                             MOT20-05.264 thr:   16 fps:  641 fps (nonskip):  453 ImgMem:  102.6MB TRTMem:  246.6MB

*/
