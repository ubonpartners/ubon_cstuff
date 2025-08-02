#ifndef __TRACK_SHARED_H
#define __TRACK_SHARED_H

#include "ctpl_stl.h"
#include <unordered_set>

typedef struct track_stream_performance_state track_stream_performance_state_t;

struct track_shared_state
{
    pthread_mutex_t lock;
    pthread_t thread_handle;
    const char *config_yaml;
    bool stop;
    int max_width, max_height;
    int num_worker_threads;
    float motiontrack_min_roi_after_skip;
    float motiontrack_min_roi_after_nonskip;
    infer_thread_t *infer_thread[INFER_THREAD_NUM_TYPES];
    model_description_t *md;
    ctpl::thread_pool *thread_pool;
    jpeg_thread_t *jpeg_thread;
    std::unordered_set<track_stream_t *> *track_stream_set;
};

struct track_stream_performance_state
{
    float dummy;
};

typedef struct track_stream_perf_data
{
    float h26x_ql_iir;
} track_stream_perf_data_t;

void track_shared_state_register_stream(track_shared_state_t *tss, track_stream_t *ts);
void track_shared_state_deregister_stream(track_shared_state_t *tss, track_stream_t *ts);
void track_stream_poll_performance_data(track_stream_t *ts, track_stream_perf_data_t *pd);

#endif