#ifndef __MOTION_TRACK_H
#define __MOTION_TRACK_H

typedef struct motion_track motion_track_t;

#include "image.h"
#include "roi.h"

typedef struct oflow_vector
{
    int16_t                         flowx;        /**< x component of flow in S10.5 format */
    int16_t                         flowy;        /**< y component of flow in S10.5 format */
} oflow_vector_t;

typedef struct of_results
{
    int grid_w, grid_h;
    oflow_vector_t *flow;
} of_results_t;

motion_track_t *motion_track_create(const char *config_yaml);
void motion_track_destroy(motion_track_t *mt);
void motion_track_add_frame(motion_track_t *mt, image_t *img);
roi_t motion_track_get_roi(motion_track_t *mt);
void motion_track_set_roi(motion_track_t *mt, roi_t roi);
of_results_t *motion_track_get_of_results(motion_track_t *mt);

#endif