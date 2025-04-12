#ifndef __DETECTIONS_H
#define __DETECTIONS_H

typedef struct kp
{
    float x, y, conf;
} kp_t;

typedef struct keypoints
{
    kp_t face_points[5];
    kp_t pose_points[17];
} keypoints_t;

typedef struct detection
{
    float x0,y0,x1,y1;
    float conf;
    unsigned short cl;
    unsigned short index;
    keypoints_t *kp;
} detection_t;

typedef struct detections
{
    int num_detections;
    int max_detections;
    detection_t det[0];
} detections_t;

#include "image.h"

detections_t *create_detections(int max_detections);
detections_t *load_detections(const char *filename);
void destroy_detections(detections_t *detections);
detection_t *detection_add_end(detections_t *detections);
void detections_nms_inplace(detections_t *detections, float iou_thr);
void detections_sort_descending_conf(detections_t *detections);
void detections_scale(detections_t *dets, float sx, float sy);
image_t *draw_detections(detections_t *dets, image_t *img);
void detection_create_keypoints(detection_t *d);
void show_detections(detections_t *dets);

#endif