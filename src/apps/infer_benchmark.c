#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>  // for std::setw and std::setprecision
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <cassert>
#include <semaphore.h>
#include <time.h>

#include "profile.h"
#include "cuda_stuff.h"
#include "log.h"
#include "display.h"
#include "jpeg.h"
#include "infer_thread.h"
#include "misc.h"

#define MAX_THREADS     32
#define MAX_IMAGES      64

#define debugf if (0) printf

typedef struct benchmark_context benchmark_context_t;
typedef struct benchmark_thread_context benchmark_thread_context_t;

typedef struct benchmark_config {
    const char *image_folder;
    int num_images;
    int num_threads;
    int width, height;
    bool use_cuda_nms;
    float run_time_seconds;
    const char *trt_model;
    const char *trt_model_config;
} benchmark_config_t;

struct benchmark_thread_context {
    benchmark_context_t *parent;
    int index;
    int num_inferences;
    int total_detections;
    pthread_t thread;
};

struct benchmark_context {
    infer_thread_t *infer_thread;
    int num_images;
    image_t *images[MAX_IMAGES];
    benchmark_thread_context_t thread_context[MAX_THREADS];
    pthread_barrier_t start_barrier;
    bool display;
    volatile bool stop;
};

typedef struct benchmark_result {
    float inferences_per_second;
    int total_detections;
    int total_inferences;
    infer_thread_stats_t infer_stats;
} benchmark_result_t;

static const char* get_last_path_part(const char* path) {
    const char* last_slash = strrchr(path, '/');
    if (last_slash) {
        return last_slash + 1;
    } else {
        return path;  // No slash found; return the whole string
    }
}

static void *benchmark_thread(void *p)
{
    benchmark_thread_context_t *tc = (benchmark_thread_context_t *)p;
    debugf("infer start %d\n",tc->index);
    cuda_thread_init();
    benchmark_context_t *bc = tc->parent;
    debugf("infer sem wait %d\n",tc->index);
    bool waited=false;

    int n = 0;
    int total_detections = 0;

    bool display=(bc->display && tc->index==0);
    double last_display_time=profile_time()-5;

    int img_cnt=0;
    while (!bc->stop)
    {
        if (n==1 && waited==false) // run one iteration as warmup before barrier wait
        {
            pthread_barrier_wait(&bc->start_barrier);
            debugf("infer thread %d running ok\n",tc->index);
            n=0;
            total_detections=0;
            waited=true;
        }

        image_t *img = bc->images[img_cnt % bc->num_images];
        img_cnt++;
        roi_t roi = { .box = {0, 0, 1, 1} };
        infer_thread_result_handle_t *h = infer_thread_infer_async(bc->infer_thread, img, roi);
        infer_thread_result_data_t d = {};
        infer_thread_wait_result(h, &d);
        if (display && 1)//profile_time()-last_display_time>1)
        {
            last_display_time=profile_time();
            image_t *out_frame=draw_detections(d.dets, img);
            show_detections(d.dets);
            //printf("img_cnt %d img %p frame %p\n",img_cnt,img,out_frame);
            display_image("test", out_frame);
            //usleep(5000000);
            destroy_image(out_frame);
        }
        total_detections += d.dets->num_detections;
        destroy_detections(d.dets);
        n++;
    }
    tc->num_inferences = n;
    tc->total_detections = total_detections;
    return NULL;
}

static void benchmark(benchmark_config_t *config, benchmark_result_t *result)
{
    benchmark_context_t bc = {};
    infer_config_t icfg = {};
    icfg.det_thr = 0.03;
    icfg.set_det_thr = true;
    icfg.nms_thr = 0.45;
    icfg.set_nms_thr = true;
    icfg.limit_max_batch=128;
    icfg.set_limit_max_batch=true;
    icfg.limit_min_width=128;
    icfg.set_limit_min_width=true;
    icfg.limit_min_height=128;
    icfg.set_limit_min_height=true;
    icfg.limit_max_width=config->width;
    icfg.set_limit_max_width=true;
    icfg.limit_max_height=config->height;
    icfg.set_limit_max_height=true;
    icfg.use_cuda_nms=config->use_cuda_nms;
    icfg.set_use_cuda_nms=true;

    pthread_barrier_init(&bc.start_barrier, NULL, config->num_threads + 1);

    bc.display=false;
    bc.infer_thread = infer_thread_start(config->trt_model, config->trt_model_config, &icfg);
    assert(bc.infer_thread!=0);
    bc.num_images = load_images_from_folder(config->image_folder, &bc.images[0],
                                            config->num_images < MAX_IMAGES ? config->num_images : MAX_IMAGES);
    if (bc.num_images==0) log_fatal("Could not load any images from folder %s (max %d)",config->image_folder, config->num_images);

    for (int i = 0; i < config->num_threads; i++) {
        bc.thread_context[i].parent = &bc;
        bc.thread_context[i].index = i;
        pthread_create(&bc.thread_context[i].thread, NULL, benchmark_thread, &bc.thread_context[i]);
    }

    pthread_barrier_wait(&bc.start_barrier);

    sleep((int)config->run_time_seconds);
    bc.stop = true;

    for (int i = 0; i < config->num_threads; i++) {
        pthread_join(bc.thread_context[i].thread, NULL);
    }

    int total_inferences = 0;
    int total_detections = 0;
    for (int i = 0; i < config->num_threads; i++) {
        total_inferences += bc.thread_context[i].num_inferences;
        total_detections += bc.thread_context[i].total_detections;
    }

    result->inferences_per_second = total_inferences / config->run_time_seconds;
    result->total_inferences = total_inferences;
    result->total_detections = total_detections;

    for (int i = 0; i < bc.num_images; i++) {
        destroy_image(bc.images[i]);
    }
    infer_thread_get_stats(bc.infer_thread, &result->infer_stats);
    infer_thread_destroy(bc.infer_thread);
    pthread_barrier_destroy(&bc.start_barrier);

}

static benchmark_config_t test_config = {
    .image_folder = "",
    .num_images = 32,
    .num_threads = 8,
    .width=640,
    .height=640,
    .use_cuda_nms=false,
    .run_time_seconds = 5.0,
    .trt_model = "/mldata/weights/trt/yolo11l-dpa-250525-dyn.trt",
    .trt_model_config = "/mldata/config/train/train_yolo_dpa_l.yaml"
};

int main(int argc, char *argv[])
{
    init_cuda_stuff();
    image_init();

    std::ostringstream oss;

    for(int images=0;images<2;images++)
    {
        if (images==0)
        {
            test_config.image_folder="/mldata/image/widerperson_100";
            test_config.num_images=32;
        }
        else
        {
            test_config.image_folder="/mldata/image/coco_100";
            test_config.num_images=32;
        }

        for(int sizes=0;sizes<3;sizes++)
        {
            if (sizes==0)
            {
                test_config.width=256;
                test_config.height=256;
            }
            else if (sizes==1)
            {
                test_config.width=416;
                test_config.height=416;
            }
            else if (sizes==2)
            {
                test_config.width=640;
                test_config.height=640;
            }
            for(int threads=1;threads<=32;threads*=2)
            {
                for(int cuda_nms=0;cuda_nms<=1;cuda_nms++)
                {
                    for(int model=0;model<2;model++)
                    {
                        if (model==0)
                            test_config.trt_model="/mldata/weights/trt/yolo11l-dpa-250525-int8.trt";
                        else
                            test_config.trt_model="/mldata/weights/trt/yolo11l-dpa-250525-dyn.trt";
                        benchmark_result_t r = {};
                        test_config.num_threads=threads;
                        test_config.use_cuda_nms=cuda_nms!=0;
                        benchmark(&test_config, &r);

                        oss << "Model " << std::setw(28) << get_last_path_part(test_config.trt_model)
                            << " Images " << std::setw(16) << get_last_path_part(test_config.image_folder)
                            << " Thr " << std::setw(2) << threads
                            << " Size " << std::setw(3) << test_config.width << "x" << std::setw(3) << test_config.height
                            << " Cu_NMS: " << cuda_nms
                            << " Avg batch: " << std::fixed << std::setprecision(2) << std::setw(6) << r.infer_stats.mean_batch_size
                            << " Inf/s: "     << std::setw(8) << r.inferences_per_second
                            << " Avg Det/f: " << std::setw(4) << (r.total_detections / (r.total_inferences+0.001))
                            << "\n";

                        std::cout << oss.str();
                    }
                }
            }
        }
    }

    return 0;
}
