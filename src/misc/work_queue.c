#include "work_queue.h"
#include "profile.h"
#include <string.h>
#include <cassert>
#include <unistd.h>
#include "log.h"

#define debugf if (0) log_error

void work_queue_init(work_queue_t *wq, ctpl::thread_pool *thread_pool,
    void *context, void (*process_item)(void *context, work_queue_item_header_t *item),
    const char *name) {
    memset(wq, 0, sizeof(work_queue_t));
    pthread_mutex_init(&wq->queue_lock, 0);
    pthread_mutex_init(&wq->work_lock, 0);
    wq->thread_pool=thread_pool;
    wq->callback_context=context;
    wq->callback=process_item;
    wq->name=name;
}

static void work_queue_process_work(int id, work_queue_t *wq);

static void work_queue_process_work(int id, work_queue_t *wq)
{
    pthread_mutex_lock(&wq->queue_lock);
    work_queue_item_header_t *job=wq->head;
    assert(job!=0);
    wq->head=job->next;
    wq->length--;
    pthread_mutex_unlock(&wq->queue_lock);
    double start_time=profile_time();
    double schedule_time=start_time-wq->schedule_time;
    wq->stats_total_schedule_time+=schedule_time;

    debugf("%s execute %p",wq->name, job);
    wq->callback(wq->callback_context, job);

    double elapsed=profile_time()-start_time;
    wq->stats_jobs_run++;
    wq->stats_total_time+=elapsed;

    if ((wq->head!=0)&&(wq->paused==false))
    {
        wq->schedule_time=profile_time();
        wq->thread_pool->push(work_queue_process_work, wq);
    }
    else
    {
        pthread_mutex_unlock(&wq->work_lock);
    }
}

int work_queue_length(work_queue_t *wq)
{
    return wq->length;
}

static void work_queue_try_start_execution(work_queue_t *wq)
{
    if (wq->paused) return;
    if (0==pthread_mutex_trylock(&wq->work_lock))
    {
        wq->schedule_time=profile_time();
        wq->thread_pool->push(work_queue_process_work, wq);
    }
}

void work_queue_add_job(work_queue_t *wq, work_queue_item_header_t *job_to_add)
{
    debugf("%s add %p l %d hwm %d", wq->name, job_to_add,wq->length,wq->stats_length_hwm);

    assert(wq->destroying==false);
    pthread_mutex_lock(&wq->queue_lock);
    work_queue_item_header_t *job=wq->head;
    job_to_add->next=0;
    if (job==0)
        wq->head=job_to_add;
    else
    {
        while(job->next!=0) job=job->next;
        job->next=job_to_add;
    }
    wq->length++;
    wq->stats_length_hwm=std::max(wq->stats_length_hwm, wq->length);
    pthread_mutex_unlock(&wq->queue_lock);
    work_queue_try_start_execution(wq);
}

void work_queue_add_job_head(work_queue_t *wq, work_queue_item_header_t *job_to_add)
{
    assert(wq->destroying==false);
    pthread_mutex_lock(&wq->queue_lock);
    work_queue_item_header_t *job=wq->head;
    wq->head=job_to_add;
    job_to_add->next=job;
    wq->length++;
    wq->stats_length_hwm=std::max(wq->stats_length_hwm, wq->length);
    pthread_mutex_unlock(&wq->queue_lock);
    work_queue_try_start_execution(wq);
}

void work_queue_pause(work_queue_t *wq, bool lock)
{
    wq->paused=true;
    if (lock) wq->locked=true;
}

void work_queue_stop(work_queue_t *wq)
{
    int iters=0;
    wq->paused=true;
    wq->stopped=true;
    // have to wait for current multi-part operations (like main pipeline)
    // to finish what they are doing
    while(1)
    {
        pthread_mutex_lock(&wq->work_lock);
        if (wq->locked==false) break;
        pthread_mutex_unlock(&wq->work_lock);
        iters++;
        if (iters>500) log_warn("wq stop: %s %d",wq->name,wq->resume_count);
        usleep(1000);
        assert(iters<10000);
    }
    wq->paused=true;
    wq->stopped=true;
}

void work_queue_resume(work_queue_t *wq)
{
    wq->locked=false;
    wq->resume_count++;
    if (wq->paused && (!wq->stopped))
    {
        wq->paused=false;
        if (wq->head!=0) work_queue_try_start_execution(wq);
    }
}

void work_queue_sync(work_queue_t *wq)
{
    // fixme : ugh - implement without sleep - need to pass dummy job with sem into queue
    int iter=0;
    while(1)
    {
        pthread_mutex_lock(&wq->work_lock);
        pthread_mutex_lock(&wq->queue_lock);
        bool empty=wq->length==0;
        pthread_mutex_unlock(&wq->queue_lock);
        pthread_mutex_unlock(&wq->work_lock);
        if (empty) break;
        iter++;
        if ((iter % 500)==0) log_warn("Wait %s : %5.1fs",wq->name, iter/1000.0);
        usleep(10000);
    }
}

void work_queue_destroy(work_queue_t *wq, void *context, void (*process_remaining_item)(void *context, work_queue_item_header_t *item))
{
    wq->destroying=true;
    pthread_mutex_lock(&wq->queue_lock);
    assert(0!=pthread_mutex_trylock(&wq->work_lock)); // should already be locked due to 'stop'
    work_queue_item_header_t *job=wq->head;
    while(job!=0)
    {
        work_queue_item_header_t *next=job->next;
        if (process_remaining_item) process_remaining_item(context, job);
        job=next;
    }
    pthread_mutex_unlock(&wq->work_lock);
    pthread_mutex_unlock(&wq->queue_lock);
    pthread_mutex_destroy(&wq->queue_lock);
    pthread_mutex_destroy(&wq->work_lock);
}

YAML::Node work_queue_get_stats(work_queue_t *wq)
{
    YAML::Node root;
    root["total_schedule_time"]=wq->stats_total_schedule_time;
    root["total_time"]=wq->stats_total_time;
    root["jobs_run"]=wq->stats_jobs_run;
    root["length_hwm"]=wq->stats_length_hwm;
    return root;
}
