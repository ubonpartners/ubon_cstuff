#include <stdint.h>
#include <stdlib.h>
#include <cassert>
#include <string.h>
#include <unistd.h>
#include "image.h"
#include "log.h"
#include "ctpl_stl.h"
#include "cuda_stuff.h"
#include "yaml_stuff.h"
#include "memory_stuff.h"
#include "jpeg_thread.h"
#include "jpeg.h"
#include "fast_histogram.h"
#include "profile.h"

#define debugf if (1) log_debug
static std::once_flag initFlag;
static block_allocator_t *jpeg_allocator;

struct jpeg_thread
{
    ctpl::thread_pool *thread_pool;
    int num_worker_threads;
    uint64_t outstanding_jpegs;
    uint64_t outstanding_jpegs_hwm;
    fast_histogram_t fh_latency;
};

struct jpeg
{
    jpeg_thread_t *jt;
    std::shared_future<bool>* completed;
    image_t *img;
    double time;
    roi_t roi;
    int max_w;
    int max_h;
    int encode_quality;
    float quality; // metric how well framed the subject is
    uint8_t *data;
    size_t data_len;
    double submit_time;
};

static void jpeg_thread_init()
{
    jpeg_allocator=block_allocator_create("jpeg", sizeof(jpeg_t));
}

static void ctpl_thread_init(int id, pthread_barrier_t *barrier)
{
    debugf("starting jpeg_thread ctpl thread %d",id);
    cuda_thread_init();
    pthread_barrier_wait(barrier);
}

jpeg_thread_t *jpeg_thread_create(const char *yaml_config)
{
    std::call_once(initFlag, jpeg_thread_init);
    YAML::Node yaml_base=yaml_load(yaml_config);
    jpeg_thread_t *jt=(jpeg_thread_t *)malloc(sizeof(jpeg_thread_t));
    assert(jt!=0);
    memset(jt, 0, sizeof(jpeg_thread_t));
    debugf("Jpeg thread create");
    fast_histogram_init(&jt->fh_latency, 0, 1.0, 1.2);
    jt->num_worker_threads=yaml_get_int_value(yaml_base["jpeg_num_worker_threads"], 2);
    debugf("%d jpeg worker threads", jt->num_worker_threads);
    // create worker threads
    jt->thread_pool=new ctpl::thread_pool(jt->num_worker_threads);
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, jt->num_worker_threads+1);
    for(int i=0;i<jt->num_worker_threads;i++) jt->thread_pool->push(ctpl_thread_init, &barrier);
    pthread_barrier_wait(&barrier);
    pthread_barrier_destroy(&barrier);
    return jt;
}

void jpeg_thread_destroy(jpeg_thread_t *jt)
{
    if (!jt) return;
    jt->thread_pool->stop();
    delete jt->thread_pool;
    free(jt);
}

static bool jpeg_encode_function(int id, jpeg_t *jpeg) {
    roi_t crop_roi;
    assert(jpeg->img!=0);
    image_t *img_cropped=image_crop_roi(jpeg->img, jpeg->roi, &crop_roi);
    size_t outsize=0;
    uint8_t *data=0;
    image_destroy(jpeg->img);
    jpeg->img=0;
    if (img_cropped!=0)
    {
        int jpeg_w, jpeg_h;
        determine_scale_size(img_cropped->width, img_cropped->height, jpeg->max_w, jpeg->max_h,
                            &jpeg_w, &jpeg_h, 0, 2, 2, false);

        if ((jpeg_w>16)&&(jpeg_h>16)) // don't try to encode too small jpeg
        {
            image_t *img_scaled=image_scale_convert(img_cropped, IMAGE_FORMAT_YUV420_HOST, jpeg_w, jpeg_h);
            data=save_jpeg_to_buffer(&outsize, img_scaled, (jpeg->encode_quality==0) ? 85 : jpeg->encode_quality);
            image_destroy(img_scaled);
        }
        else
        {
            log_error("JPEG too small %dx%d",jpeg_w,jpeg_h);
        }
        image_destroy(img_cropped);
    }

    jpeg->data=data;
    jpeg->data_len=outsize;
    double latency=profile_time()-jpeg->submit_time;
    fast_histogram_add_sample(&jpeg->jt->fh_latency, (float)latency);
    __atomic_sub_fetch(&jpeg->jt->outstanding_jpegs, 1, __ATOMIC_RELAXED);
    return true;
}

void jpeg_sync(jpeg_t *jpeg)
{
    if (jpeg->completed) jpeg->completed->wait();
}

jpeg_t *jpeg_reference(jpeg_t *jpeg)
{
    return (jpeg_t *)block_reference(jpeg);
}

jpeg_t *jpeg_thread_encode(jpeg_thread_t *jt, image_t *img, roi_t roi, int max_w, int max_h, float quality, int encode_quality)
{
    jpeg_t *jpeg=(jpeg_t *)block_alloc(jpeg_allocator, sizeof(jpeg_t));
    memset(jpeg, 0, sizeof(jpeg_t));

    jpeg->img=image_reference(img);
    jpeg->roi=roi;
    jpeg->max_w=max_w;
    jpeg->max_h=max_h;
    jpeg->encode_quality=encode_quality;
    jpeg->quality=quality;
    jpeg->time=jpeg->img->meta.time;
    jpeg->jt=jt;
    jpeg->submit_time=profile_time();
    uint64_t v=__atomic_add_fetch(&jt->outstanding_jpegs, 1, __ATOMIC_RELAXED);
    if (v>jt->outstanding_jpegs_hwm)
    {
        jt->outstanding_jpegs_hwm=v;
    }
    auto fut = std::future<bool>(jt->thread_pool->push([jpeg](int id) -> bool {
        return jpeg_encode_function(id, jpeg);
    }));
    jpeg->completed=new std::shared_future<bool>(fut.share());

    return jpeg;
}

uint8_t *jpeg_get_data(jpeg_t *jpeg, size_t *ret_size)
{
    assert(jpeg!=0);
    jpeg_sync(jpeg);

    *ret_size=jpeg->data_len;
    return jpeg->data;
}

double jpeg_get_time(jpeg_t *jpeg)
{
    return jpeg->time;
}

float jpeg_get_quality(jpeg_t *jpeg)
{
    return jpeg->quality;
}

void jpeg_destroy(jpeg_t *jpeg)
{
    if (block_reference_count(jpeg)==1)
    {
        if (jpeg->completed)
        {
            jpeg_sync(jpeg);
            delete jpeg->completed;
        }
        if (jpeg->data) free(jpeg->data);
    }
    block_free(jpeg);
}

YAML::Node jpeg_thread_stats(jpeg_thread_t *jpeg_thread)
{
    YAML::Node root;
    root["outstanding_jpegs_hwm"]=jpeg_thread->outstanding_jpegs_hwm;
    root["latency_histogram"]=fast_histogram_get_stats(&jpeg_thread->fh_latency);
    return root;
}
