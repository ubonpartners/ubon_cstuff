#ifndef __INFER_THREAD_H
#define __INFER_THREAD_H

#include "infer.h"
#include "infer_aux.h"
#include "roi.h"
#include "detections.h"
#include <yaml-cpp/yaml.h>

#define INFER_THREAD_MAX_BATCH 32

typedef enum infer_thread_type
{
    INFER_THREAD_DETECTION=0,
    INFER_THREAD_AUX_FACE=1,
    INFER_THREAD_AUX_CLIP=2,
    INFER_THREAD_AUX_FIQA=3,
    INFER_THREAD_AUX_TENSOR=4,
    INFER_THREAD_NUM_TYPES=5
} infer_thread_type_t;

static const char *infer_thread_type_names[]={
    "main_detection",
    "face_embedding",
    "clip_embedding",
    "fiqa_score",
    "audio_embedding"
};

typedef struct infer_thread infer_thread_t;

typedef struct infer_thread_result_handle infer_thread_result_handle_t;

typedef struct infer_thread_stats
{
    uint32_t batch_size_histogram[INFER_THREAD_MAX_BATCH];
    float batch_size_histogram_total_time[INFER_THREAD_MAX_BATCH];
    float batch_size_histogram_time_per_inference[INFER_THREAD_MAX_BATCH];
    uint32_t total_batches;
    uint32_t total_images;
    float total_roi_area;
    float mean_batch_size;
    float mean_roi_area;
} infer_thread_stats_t;

typedef struct infer_thread_result_data
{
    // resulting detections (must be freed with detection_destroy..)
    detection_list_t *dets;
    roi_t inference_roi; // actual used roi WRT original image
} infer_thread_result_data_t;

infer_thread_t *infer_thread_start(const char *model_trt, const char *config_yaml, infer_thread_type_t type=INFER_THREAD_DETECTION);
void infer_thread_destroy(infer_thread_t *t);
model_description_t *infer_thread_get_model_description(infer_thread_t *h);
aux_model_description_t *infer_thread_get_aux_model_description(infer_thread_t *h);
void infer_thread_configure(infer_thread_t *t, infer_config_t *config);

infer_thread_result_handle_t *infer_thread_infer_async(infer_thread_t *h, image_t *img, roi_t roi);
void infer_thread_wait_result(infer_thread_result_handle_t *h, infer_thread_result_data_t *d);

void infer_thread_infer_async_callback(infer_thread_t *h, image_t *img, roi_t roi, void (*callback)(void *context, infer_thread_result_data_t *rd), void *callback_context);
//void infer_thread_infer_async_callback_facepoints(infer_thread_t *h, image_t *img, float *fp, void (*callback)(void *context, infer_thread_result_data_t *rd), void *callback_context);
embedding_t *infer_thread_infer_embedding(infer_thread_t *h, image_t *img, kp_t *kp=0, int num_kp=0, roi_t roi=ROI_ZERO);

void infer_thread_get_stats(infer_thread_t *h, infer_thread_stats_t *s);
void infer_thread_print_stats(infer_thread_t *h);
YAML::Node infer_thread_stats_node(infer_thread_t *h);

#endif
