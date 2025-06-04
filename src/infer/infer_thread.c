// infer_thread.c

#include "infer_thread.h"
#include "cuda_stuff.h"
#include "image.h"
#include "infer.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "display.h"

static int MAX_BATCH=32;

/*
 * Internal job struct: holds the image, ROI, and result handle
 */
typedef struct infer_job {
    image_t *img;
    roi_t roi;
    infer_thread_result_handle_t *handle;
    struct infer_job *next;
} infer_job_t;

/*
 * The per-call result handle:
 * - a mutex and condition variable to wait for completion
 * - a pointer to detections when done
 * - a flag indicating completion
 */
struct infer_thread_result_handle {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    detections_t *dets;
    roi_t inference_roi;
    int done;
};

/*
 * The worker-thread context:
 * - the underlying infer_t pointer
 * - a mutex+condvar to protect the job queue
 * - a singly-linked FIFO queue of infer_job_t
 * - a flag to signal shutdown
 */
struct infer_thread {
    pthread_t thread_handle;
    infer_t *infer;
    model_description_t *md;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_cond;
    infer_job_t *job_head;
    infer_job_t *job_tail;
    int stop;
};

/*
 * Worker thread function: waits for jobs on the queue, processes them one by one,
 * signals the corresponding result handle, then loops until stop is set.
 */
static void *infer_thread_fn(void *arg)
{
    infer_thread_t *h = (infer_thread_t *)arg;
    if (!h) return NULL;

    // Initialize per‐thread CUDA / image state
    init_cuda_stuff();
    image_init();

    // Pre‐allocate fixed‐size arrays on the stack for up to MAX_BATCH jobs
    image_t          *imgs[MAX_BATCH];
    detections_t     *dets_arr[MAX_BATCH];
    infer_thread_result_handle_t *handles[MAX_BATCH];
    infer_job_t      *jobs[MAX_BATCH];
    roi_t            rois[MAX_BATCH];

    int max_batch=h->md->max_batch;
    if (max_batch>MAX_BATCH) max_batch=MAX_BATCH;

    while (1) {
        int count = 0;

        // 1) Wait until at least one job is in the queue (or stop flag is set)
        pthread_mutex_lock(&h->queue_mutex);
        while (h->job_head == NULL && !h->stop) {
            pthread_cond_wait(&h->queue_cond, &h->queue_mutex);
        }

        // 2) If stop==1 and still nothing in queue, we are done
        if (h->stop && h->job_head == NULL) {
            pthread_mutex_unlock(&h->queue_mutex);
            break;
        }

        // 3) Pop up to max_batch jobs from the head of the queue
        //    — we know job_head != NULL, so at least one iteration

        while (count < max_batch && h->job_head != NULL) {
            infer_job_t *job = h->job_head;
            h->job_head = job->next;
            if (h->job_head == NULL) {
                h->job_tail = NULL;
            }

            // Keep track of each job’s pointers so we can call infer_batch
            jobs[count]    = job;
            imgs[count]    = job->img;
            rois[count]    = job->roi;
            handles[count] = job->handle;
            count++;
        }
        pthread_mutex_unlock(&h->queue_mutex);

        // 4) Now call infer_batch(...) on the batch of `count` images
        //    We assume `infer_batch` has this prototype:
        //      void infer_batch(infer_t *inf, image_t **imgs, detections_t **dets_out, int num);
        //    and that it will allocate + return one detections_t* per input image.
        //

        image_t *img_cropped[MAX_BATCH];
        for(int i=0;i<count;i++)
        {
            img_cropped[i]=image_crop_roi(imgs[i], rois[i], &handles[i]->inference_roi);
        }
        //display_image("crop", img_cropped[0]);
        infer_batch(h->infer, img_cropped, dets_arr, count);

        for (int i = 0; i < count; ++i)
        {
            detections_unmap_roi(dets_arr[i], handles[i]->inference_roi);
            //show_detections(dets_arr[i]);
        }
        // 5) Signal each job’s result handle, passing back its own detections_t*
        for (int i = 0; i < count; ++i) {
            destroy_image(img_cropped[i]);
            destroy_image(imgs[i]);
            infer_thread_result_handle_t *rh = handles[i];
            pthread_mutex_lock(&rh->mutex);
            rh->dets = dets_arr[i];
            rh->done = 1;
            pthread_cond_signal(&rh->cond);
            pthread_mutex_unlock(&rh->mutex);

            // Free the job struct itself (we do NOT free img or dets here;
            //  - img memory is owned by the caller,
            //  - dets_arr[i] is now owned by the caller once they call wait_result).
            free(jobs[i]);
        }
        // Loop back to see if more jobs are queued (or block if none).
    }

    return NULL;
}

infer_thread_t *infer_thread_start(const char *model_trt, const char *config_yaml, infer_config_t *config)
{
    infer_thread_t *h = (infer_thread_t *)malloc(sizeof(infer_thread_t));
    assert(h != NULL);
    memset(h, 0, sizeof(infer_thread_t));

    // Create and configure the synchronous infer_t instance
    h->infer = infer_create(model_trt, config_yaml);
    infer_configure(h->infer, config);
    h->md=infer_get_model_description(h->infer);

    // Initialize queue structures
    pthread_mutex_init(&h->queue_mutex, NULL);
    pthread_cond_init(&h->queue_cond, NULL);
    h->job_head = h->job_tail = NULL;
    h->stop = 0;

    // Spawn the worker thread
    int ret = pthread_create(&h->thread_handle, NULL, infer_thread_fn, h);
    assert(ret == 0);

    return h;
}

void infer_thread_destroy(infer_thread_t *h)
{
    if (!h) return;

    // Signal the worker to stop
    pthread_mutex_lock(&h->queue_mutex);
    h->stop = 1;
    pthread_cond_signal(&h->queue_cond);
    pthread_mutex_unlock(&h->queue_mutex);

    // Wait for the thread to exit
    pthread_join(h->thread_handle, NULL);

    // Clean up any remaining jobs: notify their handles with NULL result
    pthread_mutex_lock(&h->queue_mutex);
    infer_job_t *job = h->job_head;
    while (job) {
        infer_job_t *next = job->next;
        pthread_mutex_lock(&job->handle->mutex);
        job->handle->dets = NULL;
        job->handle->done = 1;
        pthread_cond_signal(&job->handle->cond);
        pthread_mutex_unlock(&job->handle->mutex);
        free(job);
        job = next;
    }
    h->job_head = h->job_tail = NULL;
    pthread_mutex_unlock(&h->queue_mutex);

    // Destroy the synchronous infer instance
    infer_destroy(h->infer);

    // Destroy synchronization primitives
    pthread_mutex_destroy(&h->queue_mutex);
    pthread_cond_destroy(&h->queue_cond);

    free(h);
}

model_description_t *infer_thread_get_model_description(infer_thread_t *h)
{
    return h->md;
}

infer_thread_result_handle_t *infer_thread_infer_async(infer_thread_t *h, image_t *img, roi_t roi)
{
    if (!h || !img) return NULL;

    // Allocate and initialize a new result handle
    infer_thread_result_handle_t *handle = (infer_thread_result_handle_t *)malloc(sizeof(infer_thread_result_handle_t));
    assert(handle != NULL);
    pthread_mutex_init(&handle->mutex, NULL);
    pthread_cond_init(&handle->cond, NULL);
    handle->dets = NULL;
    handle->done = 0;

    // Allocate and fill in a new job
    infer_job_t *job = (infer_job_t *)malloc(sizeof(infer_job_t));
    assert(job != NULL);
    job->img = image_reference(img);
    job->roi = roi;
    job->handle = handle;
    job->next = NULL;

    // Enqueue the job
    pthread_mutex_lock(&h->queue_mutex);
    if (h->job_tail) {
        h->job_tail->next = job;
        h->job_tail = job;
    } else {
        // First job in the queue
        h->job_head = h->job_tail = job;
    }
    pthread_cond_signal(&h->queue_cond);
    pthread_mutex_unlock(&h->queue_mutex);

    return handle;
}

void infer_thread_wait_result(infer_thread_result_handle_t *handle, infer_thread_result_data_t *d)
{
    if (!handle || !d) return;

    // Wait until the worker signals completion
    pthread_mutex_lock(&handle->mutex);
    while (!handle->done) {
        pthread_cond_wait(&handle->cond, &handle->mutex);
    }
    // Copy the detection results into the user-provided struct
    d->dets = handle->dets;
    d->inference_roi=handle->inference_roi;
    pthread_mutex_unlock(&handle->mutex);

    // Clean up the handle
    pthread_mutex_destroy(&handle->mutex);
    pthread_cond_destroy(&handle->cond);
    free(handle);
}
