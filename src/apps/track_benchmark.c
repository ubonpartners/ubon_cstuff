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

typedef struct state {
    track_stream_t *ts;
    uint64_t tracked_frames;
    uint64_t tracked_frames_nonskip;
    uint64_t decoded_macroblocks;
    uint64_t decoded_frames;
    uint64_t face_embeddings;
} state_t;

typedef struct test_clip
{
    const char *friendly_name;
    const char *filename;
    float fps;
} test_clip_t;

typedef struct {
    const char *video_file_filename;
    float video_file_framerate;
    int thread_id;
    float track_framerate;
    float face_embedding_min_quality;
    unsigned int *total_tracked;
    unsigned int *total_tracked_nonskip;
    uint64_t *total_decoded_macroblocks;
    uint64_t *total_face_embeddings;
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

static void process_image(void *context, image_t *img) {
    state_t *s = (state_t *)context;
    if (keep_running)
    {
        s->decoded_macroblocks+=(img->width * img->height) / (16 * 16);
        s->decoded_frames++;
        track_stream_run_frame_time(s->ts, img);
    }
    if (1)
    {
        //display_image("Test", img);
    }
}

static void *run_track_worker(void *arg) {
    cuda_thread_init();
    thread_args_t *args = (thread_args_t *)arg;
    state_t s;
    memset(&s, 0, sizeof(state_t));

    s.ts = track_stream_create(args->tss, &s, track_result);
    track_stream_set_minimum_frame_intervals(s.ts, 1.0/args->track_framerate, 10.0);
    track_stream_enable_face_embeddings(s.ts, args->face_embedding_min_quality<1, args->face_embedding_min_quality);

    const char *filename = args->video_file_filename;
    FILE *input = fopen(filename, "rb");
    if (!input) {
        log_fatal("Failed to open input file %s", args->video_file_filename);
        return NULL;
    }


    auto stop_callback=[](void *context) {
        return !keep_running;
    };

    // decode_file continues (looping if necessary) until 'stop callback'
    decode_file(filename, &s, process_image, args->video_file_framerate, stop_callback);

    pthread_mutex_lock(args->lock);
    *args->total_tracked_nonskip += s.tracked_frames_nonskip;
    *args->total_tracked += s.tracked_frames;
    *args->total_decoded_macroblocks += s.decoded_macroblocks;
    *args->total_face_embeddings += s.face_embeddings;
    pthread_mutex_unlock(args->lock);

    fclose(input);
    track_stream_destroy(s.ts);
    return NULL;
}

typedef struct test_config
{
    const char *testset;
    char name[128];
    test_clip_t *input_clip;
    const char *yaml_config;
    int duration_sec;
    int num_threads;
    int infer_w, infer_h;
    float track_framerate;
    float face_embedding_min_quality;
    //
    float fps;
    float fps_nonskip;
    float mbps;
    float feps;
} test_config_t;

static void run_one_test(test_config_t *config)
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
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

    for (int i = 0; i < config->num_threads; ++i) {
        args[i].video_file_filename = config->input_clip->filename;
        args[i].video_file_framerate= config->input_clip->fps;
        args[i].track_framerate = config->track_framerate;
        args[i].face_embedding_min_quality=config->face_embedding_min_quality;
        args[i].thread_id = i;
        args[i].total_tracked = &total_tracked;
        args[i].total_tracked_nonskip = &total_tracked_nonskip;
        args[i].total_decoded_macroblocks = &total_decoded_macroblocks;
        args[i].total_face_embeddings=&total_face_embeddings;
        args[i].lock = &lock;
        args[i].duration_sec = config->duration_sec;
        args[i].tss = shared_state;
        pthread_create(&threads[i], NULL, run_track_worker, &args[i]);
    }

    sleep((unsigned int)config->duration_sec);
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

test_clip_t clips[]={
    {"Ind office, 720p, 7.5fps,  H265", "/mldata/video/ind_off_1280x720_7.5fps.265", 7.5},
    {"MOT20-05    720p, 6.25fps, H265", "/mldata/video/MOT20-05_1280x1080_6.25fps.265", 6.25},

    {"Ind office, 720p, 7.5fps,  H264", "/mldata/video/ind_off_1280x720_7.5fps.264", 7.5},
    {"MOT20-05    720p, 6.25fps, H264", "/mldata/video/MOT20-05_1280x1080_6.25fps.264", 6.25},
    {"UK office,  720p, 6.25fps, H264", "/mldata/video/uk_off_1280x720_6.25fps.264", 6.25},
    {"Bcam,       720p, 7.5fps,  H264", "/mldata/video/bc1_1280x720_7.5fps.264", 7.5},

    {"Ind office, 1080p, 15fps,  H264","/mldata/video/ind_off_1920x1080_15fps.264", 15.0},
    {"MOT20-05    1080p, 25fps,  H264","/mldata/video/MOT20-05_1654x1080_25fps.264", 25},
    {"UK office, 1512p, 12.5fps, H264","/mldata/video/uk_off_2688x1512_12.5fps.264", 12.5},
    {"Bcam,      1080p,  30fps,  H264","/mldata/video/bc1_1920x1080_30fps.264", 29.97},

    {"Ind office, 1080p, 15fps,  H265", "/mldata/video/ind_off_1920x1080_15fps.265", 15.0},
    {"MOT20-05    1080p, 25fps,  H265", "/mldata/video/MOT20-05_1654x1080_25fps.265", 25},
    {"UK office, 1512p, 12.5fps, H265", "/mldata/video/uk_off_2688x1512_12.5fps.265", 12.5},
    {"Bcam,      1080p,  30fps,  H265", "/mldata/video/bc1_1920x1080_30fps.265", 29.97},

};

int main(int argc, char *argv[]) {

    log_debug("ubon_cstuff version = %s", ubon_cstuff_get_version());
    init_cuda_stuff();
    log_debug("Init cuda done");
    log_debug("Initial GPU mem %f",get_process_gpu_mem(false, false));
    image_init();

    std::ostringstream oss, hdr;

    hdr   << std::setw(42)  << "Test Description" << " "
          << std::setw(20)  << "Cfg" << " "
          << std::setw(4)   << "Str" << " "
          << std::setw(8)   << "Dec" << " "
          << std::setw(5)   << "FPS" << " "
          << std::setw(5)   << "FE/S" << " "
          << std::setw(4)   << "Skp%" << " "
          << std::setw(8)   << "ImgMem" << " "
          << std::setw(8)   << "TRTMem" << " "
          << std::setw(8)   << "CudaHWM" << " "
          << std::setw(8)   << "CudaCur" << " "
          << std::setw(8)   << "GPUHWM" << " "
          << std::setw(8)   << "GPUCur" << " "
          << "\n";

    test_config_t dconfig={0};
    sprintf(dconfig.name, "default");
    dconfig.yaml_config = "/mldata/config/track/trackers/uc_reid.yaml";
    dconfig.input_clip = &clips[0];
    dconfig.duration_sec = 10;
    dconfig.track_framerate = 8;
    dconfig.face_embedding_min_quality=0.01;
    dconfig.num_threads = 16;
    dconfig.infer_w = 640;
    dconfig.infer_h = 640;

    test_config_t config[64];
    for(int i=0;i<64;i++) config[i]=dconfig;
    int nconfig;

    for(int i=0;i<8;i++)
    {
        static float q[4]={1.0, 0.1, 0.01, 0};
        config[nconfig].testset="Vary face embeddings";
        config[nconfig].input_clip = &clips[i/4];
        config[nconfig].face_embedding_min_quality=q[i%4];
        sprintf(config[nconfig++].name, "FaceQ %s:%.3f",clips[i/4].friendly_name,q[i%4]);
    }

    {
        config[nconfig].testset="Single test UK OF 1512p H265";
        config[nconfig].input_clip = &clips[12];
        config[nconfig].num_threads=1;
        sprintf(config[nconfig++].name, "%s",clips[12].friendly_name);
    }

    for(int i=0;i<4;i++)
    {
        config[nconfig].testset="Vary tracker";
        config[nconfig].input_clip = &clips[i/2];
        if ((i&1)==1) config[nconfig].yaml_config = "/mldata/config/track/trackers/uc_bytetrack.yaml";
        sprintf(config[nconfig++].name, "%s:%s",((i&1)==1) ? "Bytetrack" : "UC-Reid", clips[i/2].friendly_name);
    }

    for(int i=0;i<14;i++)
    {
        config[nconfig].testset="Vary input video";
        config[nconfig].input_clip = &clips[i];
        sprintf(config[nconfig++].name, "%s",clips[i].friendly_name);
    }

    for(int i=0;i<5;i++)
    {
        int sizes[]={640, 512, 416, 320, 256};
        int size=sizes[i];
        config[nconfig].testset="Vary inference max resolution";
        config[nconfig].infer_w = size;
        config[nconfig].infer_h = size;
        config[nconfig].num_threads = 16;
        sprintf(config[nconfig++].name, "Infer max @ %dx%d", size, size);
    }

    /*for(int i=0;i<4;i++)
    {
        config[nconfig].testset="Vary quantization";
        int quant=i/2;
        int clip=i%2;
        config[nconfig].input_clip = &clips[clip];
        config[nconfig].yaml_config = (quant == 0)
            ? "/mldata/config/track/trackers/uc_test.yaml"
            : "/mldata/config/track/trackers/uc_test_fp16.yaml";
        sprintf(config[nconfig++].name, " %s %s", clips[clip].friendly_name, (quant==1) ? "fp16" : "int8");
    }*/
    for(int i=0;i<6;i++)
    {
        config[nconfig].testset="Vary number of streams";
        config[nconfig].num_threads=1<<i;
        sprintf(config[nconfig++].name, "Streams: %2d", 1<<i);
    }


    for(int t=0;t<nconfig;t++)
    {
        /*config.yaml_config = (c == 0)
            ? "/mldata/config/track/trackers/uc_test.yaml"
            : "/mldata/config/track/trackers/uc_test_fp16.yaml";
        config.video_file_filename = clips[tf].name;
        config.video_file_framerate = clips[tf].fps;
        config.duration_sec = 30;
        config.track_framerate = 8;
        config.num_threads = 1 << th;*/
        test_config_t *this_config=&config[t];
        run_one_test(this_config);

        auto format_mb = [](double bytes) -> double {
            return bytes / 1e6;  // Convert bytes to MB
        };

        double app_curr = get_cuda_mem(false, false, false);
        double app_hwm = get_cuda_mem(false, true, true);
        cuda_flush();
        double gpu_mem_curr = get_process_gpu_mem(false, false);
        double gpu_mem_hwm = get_process_gpu_mem(true, true);
        int skip_percent = (int)(100 * (1.0 - this_config->fps_nonskip / (this_config->fps + 0.001)));

        if (t==0 || strcmp(this_config->testset, config[t-1].testset) != 0) {
            oss << "\n== " << this_config->testset << " ==\n";
            oss << hdr.str();
        }
        oss     << std::setw(42) << this_config->name
                << " " << std::setw(20) << get_last_path_part(this_config->yaml_config)
                << " " << std::setw(4)  << this_config->num_threads
                << " " << std::setw(8)  << ((int)(this_config->mbps/3600.0))
                << " " << std::setw(5)  << ((int)this_config->fps)
                << " " << std::setw(5)  << ((int)this_config->feps)
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

Video                                  bc1.264 thr:    1 fps:  329 fps (nonskip):  329 ImgMem:   15.6MB TRTMem:  246.6MB Cuda HWM:  305.3MB Cuda Curr:    0.0MB
Video                                  bc1.264 thr:    2 fps:  359 fps (nonskip):  359 ImgMem:   23.5MB TRTMem:  246.6MB Cuda HWM:  313.2MB Cuda Curr:    0.0MB
Video                                  bc1.264 thr:    4 fps:  547 fps (nonskip):  547 ImgMem:   47.0MB TRTMem:  246.6MB Cuda HWM:  336.7MB Cuda Curr:    0.0MB
Video                                  bc1.264 thr:    8 fps:  676 fps (nonskip):  676 ImgMem:   93.9MB TRTMem:  246.6MB Cuda HWM:  383.8MB Cuda Curr:    0.0MB
Video                                  bc1.264 thr:   16 fps:  680 fps (nonskip):  680 ImgMem:  149.4MB TRTMem:  246.6MB Cuda HWM:  438.6MB Cuda Curr:    0.0MB
Video                             MOT20-05.264 thr:    1 fps:  166 fps (nonskip):  166 ImgMem:   14.0MB TRTMem:  246.6MB Cuda HWM:  309.3MB Cuda Curr:    0.0MB
Video                             MOT20-05.264 thr:    2 fps:  286 fps (nonskip):  286 ImgMem:   27.5MB TRTMem:  246.6MB Cuda HWM:  322.8MB Cuda Curr:    0.0MB
Video                             MOT20-05.264 thr:    4 fps:  390 fps (nonskip):  390 ImgMem:   50.2MB TRTMem:  246.6MB Cuda HWM:  345.6MB Cuda Curr:    0.0MB
Video                             MOT20-05.264 thr:    8 fps:  515 fps (nonskip):  515 ImgMem:   89.1MB TRTMem:  246.6MB Cuda HWM:  380.8MB Cuda Curr:    0.0MB
Video                             MOT20-05.264 thr:   16 fps:  571 fps (nonskip):  571 ImgMem:  145.8MB TRTMem:  246.6MB Cuda HWM:  440.4MB Cuda Curr:    0.0MB

*/
