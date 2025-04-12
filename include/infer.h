#ifndef __INFER_H
#define __INFER_H

typedef struct infer infer_t;

#include "image.h"
#include "detections.h"

infer_t *infer_create(const char *model);
void infer_destroy(infer_t *inf);
detections_t *infer(infer_t *inf, image_t *img);

#endif

