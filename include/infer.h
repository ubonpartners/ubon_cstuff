#ifndef __INFER_H
#define __INFER_H

typedef struct infer infer_t;

#include "image.h"
#include "detections.h"

infer_t *infer_create(const char *model_trt, const char *config_yaml);
void infer_destroy(infer_t *inf);
detections_t *infer(infer_t *inf, image_t *img);
void infer_batch(infer_t *inf, image_t **img, detections_t **dets, int num);

#endif
