#ifndef __NVOF_H
#define __NVOF_H

#include <stdint.h>

typedef struct nvof nvof_t;

#include "image.h"

typedef struct flow_vector
{
    int16_t                         flowx;        /**< x component of flow in S10.5 format */
    int16_t                         flowy;        /**< y component of flow in S10.5 format */
} flow_vector_t;

typedef struct nvof_result
{
    int grid_w, grid_h;
    uint8_t *costs;
    flow_vector_t *flow;
} nvof_results_t;

nvof_t *nvof_create(void *context, int w, int h);
void nvof_destroy(nvof_t *v);
nvof_results_t *nvof_execute(nvof_t *v, image_t *img);
void nvof_reset(nvof_t *n);
void nvof_set_no_motion(nvof_t *v);

#endif
