#ifndef __INFER_THREAD_H
#define __INFER_THREAD_H

#include "infer.h"
#include "roi.h"
#include "detections.h"

typedef struct infer_thread infer_thread_t;

typedef struct infer_thread_result_handle infer_thread_result_handle_t;

typedef struct infer_thread_result_data
{
    // resulting detections (must be freed with detection_destroy..)
    detections_t *dets;
    // execution into/stats
    float queue_time;
    float inference_time;
    roi_t inference_roi; // actual used roi WRT original image
} infer_thread_result_data_t;

infer_thread_t *infer_thread_start(const char *model_trt, const char *config_yaml, infer_config_t *config);
void infer_thread_destroy(infer_thread_t *t);
model_description_t *infer_thread_get_model_description(infer_thread_t *h);

infer_thread_result_handle_t *infer_thread_infer_async(infer_thread_t *h, image_t *img, roi_t roi);
void infer_thread_wait_result(infer_thread_result_handle_t *h, infer_thread_result_data_t *d);


#endif
