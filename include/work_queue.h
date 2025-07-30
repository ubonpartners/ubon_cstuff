#ifndef __WORK_QUEUE_H
#define __WORK_QUEUE_H

#include <yaml-cpp/yaml.h>
#include "ctpl_stl.h"

typedef struct work_queue work_queue_t;
typedef struct work_queue_item_header work_queue_item_header_t;

struct work_queue_item_header
{
    work_queue_item_header_t *next;
};

struct work_queue
{
    work_queue_item_header_t *head;
    pthread_mutex_t queue_lock;
    pthread_mutex_t work_lock;
    ctpl::thread_pool *thread_pool;
    bool paused;
    bool locked;
    bool stopped;
    bool destroying;
    uint32_t length;
    uint32_t resume_count;
    const char *name;
    double schedule_time;
    void *callback_context;
    void (*callback)(void *context, work_queue_item_header_t *item);
    double stats_total_schedule_time;
    double stats_total_time;
    uint32_t stats_length_hwm;
    uint32_t stats_jobs_run;
};

void work_queue_init(work_queue_t *wq, ctpl::thread_pool *thread_pool,
    void *context, void (*process_item)(void *context, work_queue_item_header_t *item),
    const char *name);
void work_queue_add_job(work_queue_t *wq, work_queue_item_header_t *job);
void work_queue_add_job_head(work_queue_t *wq, work_queue_item_header_t *job);
void work_queue_pause(work_queue_t *wq, bool lock=false);
void work_queue_stop(work_queue_t *wq);
void work_queue_resume(work_queue_t *wq);
void work_queue_sync(work_queue_t *wq);
int work_queue_length(work_queue_t *wq);
void work_queue_destroy(work_queue_t *wq, void *context, void (*process_remaining_item)(void *context, work_queue_item_header_t *item));
YAML::Node work_queue_get_stats(work_queue_t *wq);

#endif
