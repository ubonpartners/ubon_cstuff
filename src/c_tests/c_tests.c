#include "simple_decoder.h"
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cuda_stuff.h"
#include "display.h"
#include "webcam.h"
#include "infer.h"
#include "jpeg.h"
#include "dataset.h"
#include "nvof.h"
#include "c_tests.h"
#include "misc.h"


static uint32_t test_convert_3(image_t *img, image_format_t fmt1, image_format_t fmt2,  image_format_t fmt3)
{
    image_t *converted1=image_convert(img, fmt1);
    image_t *converted2=image_convert(converted1, fmt2);
    image_t *converted3=image_convert(converted2, fmt3);
    uint32_t hash=image_hash(converted3);
    destroy_image(converted1);
    destroy_image(converted2);
    destroy_image(converted3);
    return hash;
}

static uint32_t test_convert_yuv420(image_t *img) {return test_convert_3(img, IMAGE_FORMAT_YUV420_DEVICE,IMAGE_FORMAT_YUV420_DEVICE,IMAGE_FORMAT_YUV420_DEVICE);}
static uint32_t test_convert_yuv420_mono(image_t *img) {return test_convert_3(img, IMAGE_FORMAT_YUV420_DEVICE,IMAGE_FORMAT_MONO_DEVICE,IMAGE_FORMAT_YUV420_DEVICE);}


static uint32_t test_scale_generic(image_t *img, int w, int h, int w2, int h2, image_format_t fmt)
{
    image_t *converted=image_convert(img, fmt);
    image_sync(converted);
    image_t *scaled=image_scale(img, w, h);
    image_t *scaled2=image_scale(scaled, w2, h2);
    uint32_t hash=image_hash(scaled2);
    destroy_image(scaled);
    destroy_image(scaled2);
    destroy_image(converted);
    return hash;
}

static uint32_t test_scale_yuv420_1280(image_t *img) {return test_scale_generic(img, 1280, 720, 512, 240, IMAGE_FORMAT_YUV420_DEVICE);}
static uint32_t test_scale_yuv420_64(image_t *img) {return test_scale_generic(img, 64, 64, 512, 512, IMAGE_FORMAT_YUV420_DEVICE);}
static uint32_t test_scale_mono_64(image_t *img) {return test_scale_generic(img, 64, 64, 512, 512, IMAGE_FORMAT_MONO_DEVICE);}


typedef struct {
    uint32_t (*test)(image_t *img);
    image_t *img;
    uint32_t *hashes;
    int start;
    int end;
} thread_arg_t;

static void *test_thread_fn(void *arg)
{
    thread_arg_t *targ = (thread_arg_t *)arg;
    //log_debug("Hello from test thread");
    for (int i = targ->start; i < targ->end; i++) {
        targ->hashes[i] = targ->test(targ->img);
    }
    return NULL;
}

static void run_one_test(const char *txt, uint32_t (*test)(image_t *img), int runs, image_t *img, int num_threads)
{
    pthread_t threads[num_threads];
    thread_arg_t thread_args[num_threads];
    uint32_t *hashes = (uint32_t *)malloc(sizeof(uint32_t) * runs);

    int chunk = runs / num_threads;
    int remainder = runs % num_threads;

    double start_time=time_now_sec();
    int start = 0;
    for (int i = 0; i < num_threads; i++) {
        int end = start + chunk + (i < remainder ? 1 : 0);
        thread_args[i] = (thread_arg_t){
            .test = test,
            .img = img,
            .hashes = hashes,
            .start = start,
            .end = end
        };
        pthread_create(&threads[i], NULL, test_thread_fn, &thread_args[i]);
        start = end;
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    double elapsed_time=time_now_sec()-start_time;

    int fails = 0;
    for (int i = 1; i < runs; i++) {
        if (hashes[i] != hashes[i - 1]) {
            fails++;
        }
    }
    printf("%40s FAILS:%d; %f iterations/sec\n", txt, fails, (float)(runs/elapsed_time));
    free(hashes);
}

int run_all_c_tests()
{
    printf("Running all tests....\n");
    
    cuda_set_sync_mode(false, false);

    image_t *img_base=load_jpeg("/mldata/image/arrest2.jpg");
    image_t *img=image_scale(img_base, 1280, 720);

    //display_image("test", img);
    //usleep(5000);
    
    int runs=5000;
    int threads=16;
    run_one_test("test convert yuv420_device", test_convert_yuv420, runs, img, threads);
    run_one_test("test convert yuv420_mono_device", test_convert_yuv420_mono, runs, img, threads);
    run_one_test("test scale yuv420_device 1280x720", test_scale_yuv420_1280, runs, img, threads);
    run_one_test("test scale yuv420_device 64x64", test_scale_yuv420_64, runs, img, threads);
    run_one_test("test scale mono_device 64x64", test_scale_mono_64, runs, img, threads);
    return 0;
}