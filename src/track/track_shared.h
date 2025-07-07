#ifndef __TRACK_SHARED_H
#define __TRACK_SHARED_H

#include "ctpl_stl.h"

struct track_shared_state
{
    const char *config_yaml;
    int max_width, max_height;
    int num_worker_threads;
    float motiontrack_min_roi_after_skip;
    float motiontrack_min_roi_after_nonskip;
    infer_thread_t *infer_thread[INFER_THREAD_NUM_TYPES];
    const char *tracker_type;
    model_description_t *md;
    ctpl::thread_pool *thread_pool;
};

#endif