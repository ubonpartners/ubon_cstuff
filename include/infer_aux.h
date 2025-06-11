#ifndef __INFER_AUX_H
#define __INFER_AUX_H

#include <string>
#include <vector>

typedef struct infer_aux infer_aux_t;

#include "image.h"

infer_aux_t *infer_aux_create(const char *model_trt);
void infer_aux_destroy(infer_aux_t *inf);
float *infer_aux_batch(infer_aux_t *inf, image_t **img, float *kp, int n);

#endif
