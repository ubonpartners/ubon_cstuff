#ifndef __INFER_AUX_H
#define __INFER_AUX_H

#include <string>
#include <vector>

typedef struct infer_aux infer_aux_t;
typedef struct aux_model_description aux_model_description_t;

#include "image.h"
#include "embedding.h"

struct aux_model_description
{
    int embedding_size;
    int embedding_size_2nd_output;
    int max_w,max_h,max_batch;
    int input_dims, input_ch, input_w, input_h;
    bool input_fp16, output_fp16, output2_fp16; // input/output formats for model
    const char *engineInfo;
};

infer_aux_t *infer_aux_create(const char *model_trt, const char *config_yaml);
void infer_aux_destroy(infer_aux_t *inf);
float *infer_aux_batch(infer_aux_t *inf, image_t **img, float *kp, int n); // pass kp=0 if input images already aligned
void infer_aux_batch(infer_aux_t *inf, image_t **img, embedding_t **ret_emb, float *kp, int n);
void infer_aux_batch_roi(infer_aux_t *inf, image_t **img, embedding_t **ret_emb, roi_t *rois, int n);
void infer_aux_batch_tensor(infer_aux_t* inf, image_t **images, embedding_t **ret_emb, int n);

aux_model_description_t *infer_aux_get_model_description(infer_aux_t *inf);

#endif
