#include "work_queue.h"
#include "profile.h"
#include <string.h>
#include <cassert>
#include <unistd.h>
#include "log.h"

#define debugf if (0) log_error

void work_queue_init(work_queue_t *wq, ctpl::thread_pool *thread_pool,
    void *context, void (*process_item)(void *context, work_queue_item_header_t *item),
    const char *name, const char *name2) {
    memset(wq, 0, sizeof(work_queue_t));
    pthread_mutex_init(&wq->lock, 0);
    wq->thread_pool=thread_pool;
    wq->callback_context=context;
    wq->callback=process_item;
    wq->name=name;
    wq->name2=name2;
    wq->backpressure_length=100;
}

static void work_queue_process_work(int id, work_queue_t *wq);

static void work_queue_pause_internal(work_queue_t *wq)
{
    assert(0!=pthread_mutex_trylock(&wq->lock));
    if (wq->paused==false) wq->stats_pause_count++;
    wq->paused=true;
}

static void work_queue_try_start_execution(work_queue_t *wq)
{
    assert(0!=pthread_mutex_trylock(&wq->lock));
    debugf("[%s:%s] try start execution: length=%d paused=%d (PC %d RC %d)", wq->name,wq->name2, wq->length,wq->paused,wq->stats_pause_count,wq->stats_resume_count);
    if (wq->length==0) return;
    if (wq->paused || wq->stopped || wq->destroying) return;
    if (wq->executing==false)
    {
        debugf("[%s] pushed work run",wq->name);
        wq->schedule_time=profile_time();
        wq->executing=true;
        wq->thread_pool->push(work_queue_process_work, wq);
    }
    else
    {
        debugf("[%s] no push as already executing", wq->name);
    }
}

static void work_queue_resume_internal(work_queue_t *wq)
{
    wq->locked=false;
    if (wq->paused==true) wq->stats_resume_count++;
    wq->paused=false;
    work_queue_try_start_execution(wq);
}

static void work_queue_process_work(int id, work_queue_t *wq)
{
    pthread_mutex_lock(&wq->lock);
    work_queue_item_header_t *job=wq->head;
    assert(job!=0);
    wq->head=job->next;
    assert(true==wq->executing);
    bool need_resume=(wq->length==wq->backpressure_length);
    debugf("[%s] *** start execute %p %d %d",wq->name, job,wq->length, wq->backpressure_length);
    wq->length--;
    pthread_mutex_unlock(&wq->lock);

    if (need_resume)
    {
        for(int i=0;i<wq->num_wq_backpressure;i++)
        {
            work_queue_t *wq_bp=wq->wq_backpressure[i];
            pthread_mutex_lock(&wq_bp->lock);
            debugf("[%s] WQ resuming %s (with len=%d, paused=%d)",
                wq->name,
                wq_bp->name,
                wq_bp->length,
                wq_bp->paused);

            work_queue_resume_internal(wq_bp);
            pthread_mutex_unlock(&wq_bp->lock);
        }
    }

    double start_time=profile_time();
    double schedule_time=start_time-wq->schedule_time;
    wq->stats_total_schedule_time+=schedule_time;

    debugf("[%s] execute %p",wq->name, job);
    wq->callback(wq->callback_context, job);
    debugf("[%s] execute callback done%p",wq->name, job);
    double elapsed=profile_time()-start_time;

    pthread_mutex_lock(&wq->lock);
    wq->stats_jobs_run++;
    wq->stats_total_time+=elapsed;
    wq->executing=false;
    debugf("[%s] end of execute paused=%d",wq->name,wq->paused);
    work_queue_try_start_execution(wq);
    pthread_mutex_unlock(&wq->lock);
}

int work_queue_length(work_queue_t *wq)
{
    return wq->length;
}

void work_queue_add_job(work_queue_t *wq, work_queue_item_header_t *job_to_add, bool head)
{
    pthread_mutex_lock(&wq->lock);
    debugf("[%s] add job %p l %d hwm %d", wq->name, job_to_add,wq->length,wq->stats_length_hwm);
    assert(wq->destroying==false);
    work_queue_item_header_t *job=wq->head;

    if (head==false) {
        job_to_add->next=0;
        if (job==0)
            wq->head=job_to_add;
        else
        {
            while(job->next!=0) job=job->next;
            job->next=job_to_add;
        }
    }
    else {
        wq->head=job_to_add;
        job_to_add->next=job;
    }
    wq->length++;
    wq->stats_length_hwm=std::max(wq->stats_length_hwm, wq->length);

    // backpressure
    bool need_backpressure=(wq->length>=wq->backpressure_length);
    debugf("[%s] *** (len %3d) Applying backpressure",wq->name,wq->length);
    //pthread_mutex_unlock(&wq->lock);

    if (need_backpressure)
    {
        for(int i=0;i<wq->num_wq_backpressure;i++)
        {
            work_queue_t *wq_bp=wq->wq_backpressure[i];
            pthread_mutex_lock(&wq_bp->lock);

            debugf("[%s] (len %3d) pausing %s (existing %d)",wq->name, wq->length,
                wq->wq_backpressure[i]->name,
                wq->wq_backpressure[i]->paused);
            if (wq_bp->paused==false) wq_bp->stats_pause_count++;
            wq_bp->paused=true;
            pthread_mutex_unlock(&wq_bp->lock);
        }
    }

    //pthread_mutex_lock(&wq->lock);
    work_queue_try_start_execution(wq);
    pthread_mutex_unlock(&wq->lock);
}

void work_queue_pause(work_queue_t *wq, bool lock)
{
    pthread_mutex_lock(&wq->lock);
    work_queue_pause_internal(wq);
    if (lock) wq->locked=true;
    pthread_mutex_unlock(&wq->lock);
}

void work_queue_stop(work_queue_t *wq)
{
    int iters=0;
    pthread_mutex_lock(&wq->lock);
    wq->paused=true;
    wq->stopped=true;
    pthread_mutex_unlock(&wq->lock);
    // have to wait for current multi-part operations (like main pipeline)
    // to finish what they are doing
    while(1)
    {
        pthread_mutex_lock(&wq->lock);
        if ((wq->locked==false)&&(wq->executing==false)) break;
        pthread_mutex_unlock(&wq->lock);
        iters++;
        if ((iters%1000)==0)
        {
            log_warn("wq stop: %s:%s %d l %d jr %d",wq->name,wq->name2,wq->stats_resume_count,wq->length,wq->stats_jobs_run);
        }
        usleep(1000);
        assert(iters<20000);
    }
    wq->paused=true;
    wq->stopped=true;
    pthread_mutex_unlock(&wq->lock);
}

void work_queue_resume(work_queue_t *wq)
{
    pthread_mutex_lock(&wq->lock);
    work_queue_resume_internal(wq);
    pthread_mutex_unlock(&wq->lock);
}

void work_queue_sync(work_queue_t *wq)
{
    // fixme : ugh - implement without sleep - need to pass dummy job with sem into queue
    int iter=0;
    double sync_start_time=profile_time();
    double last_sync_check_time=sync_start_time;
    uint32_t last_job_run=0;
    while(1)
    {
        pthread_mutex_lock(&wq->lock);
        assert(wq->destroying==false);
        assert(wq->stopped==false);
        bool empty=wq->length==0 && wq->executing==false;
        uint32_t jobs_run=wq->stats_jobs_run;
        pthread_mutex_unlock(&wq->lock);
        if (empty) break;
        iter++;
        double time=profile_time();
        if (time-last_sync_check_time>1.0)
        {
            last_sync_check_time=time;
            int jobs_run_delta=jobs_run-last_job_run;
            last_job_run=jobs_run;
            if (jobs_run_delta!=0)
            {
                //log_warn("work_queue_sync Wait %s : Still running",wq->name);
                ;
            }
            else
            {
                log_warn("work_queue_sync Wait %s:%s : %5.1fs (%d entries, stopped %d destroying %d ex %d JR %d PC %d RC %d)",
                                            wq->name,wq->name2, (time-sync_start_time), wq->length, wq->stopped, wq->destroying,
                                            wq->executing, wq->stats_jobs_run, wq->stats_pause_count, wq->stats_resume_count);
                pthread_mutex_lock(&wq->lock);
                work_queue_try_start_execution(wq);
                pthread_mutex_unlock(&wq->lock);
            }
        }
        usleep(10000);
    }
}

void work_queue_destroy(work_queue_t *wq, void *context, void (*process_remaining_item)(void *context, work_queue_item_header_t *item))
{
    wq->destroying=true;
    assert(wq->stopped==true);

    pthread_mutex_lock(&wq->lock);
    work_queue_item_header_t *job=wq->head;
    while(job!=0)
    {
        work_queue_item_header_t *next=job->next;
        if (process_remaining_item) process_remaining_item(context, job);
        job=next;
    }
    pthread_mutex_unlock(&wq->lock);
    pthread_mutex_destroy(&wq->lock);
}

void work_queue_set_backpressure_length(work_queue_t *wq, int length)
{
    wq->backpressure_length=length;
}

void work_queue_set_backpressure_queue(work_queue_t *wq, work_queue_t *wq_feeding)
{
    assert(wq->num_wq_backpressure<MAX_WQ_BACKPRESSURE);
    wq->wq_backpressure[wq->num_wq_backpressure++]=wq_feeding;
}

bool work_queue_needs_backpressure(work_queue_t *wq)
{
    return wq->length>=wq->backpressure_length;
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
