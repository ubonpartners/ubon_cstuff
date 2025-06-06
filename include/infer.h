#ifndef __INFER_H
#define __INFER_H

#include <string>
#include <vector>

typedef struct infer infer_t;
typedef struct model_description model_description_t;

#include "image.h"
#include "detections.h"

struct model_description
{
    std::vector<std::string> class_names;
    std::vector<std::string> person_attribute_names;
    int num_classes;
    int num_person_attributes;
    int num_keypoints;
    int max_batch;
    int reid_vector_len;
    bool input_is_fp16, output_is_fp16; // input/output formats for model
    int min_w, max_w;  // min,max model input width
    int min_h, max_h;  // min,max model input height
    int model_output_dims[3]; // output tensor dimensions
    const char *engineInfo;
};

typedef struct infer_config
{
    float det_thr; // overall detection threshold for all classes
    float nms_thr;
    bool use_cuda_nms;
    bool fuse_face_person;
    int limit_max_batch;
    int limit_min_width;
    int limit_min_height;
    int limit_max_width;
    int limit_max_height;
    int max_detections;
    bool set_det_thr;
    bool set_nms_thr;
    bool set_use_cuda_nms;
    bool set_limit_max_batch;
    bool set_limit_min_width;
    bool set_limit_min_height;
    bool set_limit_max_width;
    bool set_limit_max_height;
    bool set_max_detections;
    bool set_fuse_face_person;
} infer_config_t;

infer_t *infer_create(const char *model_trt, const char *config_yaml);
void infer_destroy(infer_t *inf);
void infer_configure(infer_t *inf, infer_config_t *config);
model_description_t *infer_get_model_description(infer_t *inf);
void infer_print_model_description(model_description_t *md);
detections_t *infer(infer_t *inf, image_t *img);
void infer_batch(infer_t *inf, image_t **img, detections_t **dets, int num);

#endif
