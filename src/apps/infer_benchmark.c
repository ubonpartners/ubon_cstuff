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
#include "platform_stuff.h"
#include "profile.h"
#include "cuda_stuff.h"
#include "log.h"
#include "display.h"
#include "jpeg.h"
#include "infer_thread.h"
#include "misc.h"

#define MAX_THREADS     64
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
            image_t *out_frame=detection_list_draw(d.dets, img);
            detection_list_show(d.dets);
            //printf("img_cnt %d img %p frame %p\n",img_cnt,img,out_frame);
            display_image("test", out_frame);
            //usleep(5000000);
            image_destroy(out_frame);
        }
        total_detections += d.dets->num_detections;
        detection_list_destroy(d.dets);
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
    bc.infer_thread = infer_thread_start(config->trt_model, config->trt_model_config);
    infer_thread_configure(bc.infer_thread, &icfg);
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
        image_destroy(bc.images[i]);
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
    .use_cuda_nms=true,
    .run_time_seconds = 8.0,
    .trt_model = "/mldata/models/v8/trt/yolo11l-v8r-130825-int8.trt",
    .trt_model_config = "/mldata/config/train/train_yolo_v8_l.yaml"
};

int main(int argc, char *argv[])
{
    init_cuda_stuff();
    image_init();

    bool is_jetson=platform_is_jetson();

    test_config.num_threads=(is_jetson) ? 8 : 64; // max number of threads lower on jetson

    // Provide a default image set if not specified
    if (test_config.image_folder == NULL || strlen(test_config.image_folder) == 0) {
        test_config.image_folder = "/mldata/image/widerperson_100";
        test_config.num_images = 32;
    }

    std::ostringstream table;

    auto append_header = [&](std::ostringstream &out) {
        out << std::left
            << std::setw(30) << "Model"
            << std::setw(18) << "Images"
            << std::right
            << std::setw(8)  << "Thr"
            << std::setw(10) << "Size"
            << std::setw(9)  << "CuNMS"
            << std::setw(12) << "AvgBatch"
            << std::setw(12) << "Inf/s"
            << std::setw(12) << "AvgDet/f"
            << "\n";
        out << std::string(30+18+8+10+9+12+12+12, '-') << "\n";
    };

    auto append_row = [](std::ostringstream &out, const benchmark_config_t &cfg, const benchmark_result_t &r) {
        out << std::left
            << std::setw(30) << get_last_path_part(cfg.trt_model)
            << std::setw(18) << get_last_path_part(cfg.image_folder)
            << std::right
            << std::setw(8)  << cfg.num_threads
            << std::setw(10) << (std::to_string(cfg.width) + "x" + std::to_string(cfg.height))
            << std::setw(9)  << (cfg.use_cuda_nms ? "1" : "0")
            << std::fixed << std::setprecision(2)
            << std::setw(12) << r.infer_stats.mean_batch_size
            << std::setprecision(1)
            << std::setw(12) << r.inferences_per_second
            << std::setprecision(2)
            << std::setw(12) << (r.total_detections / (r.total_inferences + 0.001f))
            << "\n";
    };

    auto run_once = [&](const benchmark_config_t &cfg) {
        benchmark_result_t r = {};
        benchmark((benchmark_config_t*)&cfg, &r);
        append_row(table, cfg, r);
        // Print the entire accumulated table to avoid interleaving with other debug output
        std::cout << table.str() << std::flush;
    };

    const benchmark_config_t base = test_config;

    // Threads sweep: 1..max doubling
    table.str(""); table.clear();
    append_header(table);
    table << "Baseline\n";
    {
        benchmark_config_t cfg = base;
        run_once(cfg);
        //exit(0);
        run_once(cfg);
        run_once(cfg);
    }

    // Threads sweep: 1..max doubling
    table << "\nThreads Sweep\n";
    int max_threads = (is_jetson) ? 16 : MAX_THREADS;
    for (int thr = 1; thr <= max_threads; thr *= 2) {
        benchmark_config_t cfg = base;
        cfg.num_threads = thr;
        run_once(cfg);
    }

    // Image size sweep
    table << "\nImage Size Sweep\n";
    const int sizes[][2] = { {192,192},{256,256}, {384,384}, {416,416}, {512,512}, {640,640} };
    for (auto &sz : sizes) {
        benchmark_config_t cfg = base;
        cfg.width = sz[0];
        cfg.height = sz[1];
        run_once(cfg);
    }

    // Model sweep
    //table.str(""); table.clear();
    table << "\nModel Sweep\n";
    //append_header(table);
    {
        benchmark_config_t cfg = base;
        cfg.trt_model = "/mldata/models/v8/trt/yolo11l-v8r-130825-int8.trt";
        run_once(cfg);
    }
    {
        benchmark_config_t cfg = base;
        cfg.trt_model = "/mldata/models/v8/trt/yolo11l-v8r-130825-fp16.trt";
        run_once(cfg);
    }

    // CUDA NMS sweep
    //table.str(""); table.clear();
    table << "\nCUDA NMS Sweep\n";
    //append_header(table);
    for (int nms = 0; nms <= 1; ++nms) {
        benchmark_config_t cfg = base;
        cfg.use_cuda_nms = (nms != 0);
        run_once(cfg);
    }

    // Image set sweep
    //table.str(""); table.clear();
    table << "\nImage Set Sweep\n";
    //append_header(table);
    {
        benchmark_config_t cfg = base;
        cfg.image_folder = "/mldata/image/widerperson_100";
        cfg.num_images = 32;
        run_once(cfg);
    }
    {
        benchmark_config_t cfg = base;
        cfg.image_folder = "/mldata/image/coco_100";
        cfg.num_images = 32;
        run_once(cfg);
    }

    return 0;
}
