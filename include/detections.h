#ifndef __DETECTIONS_H
#define __DETECTIONS_H

#include <stdint.h>

#define DETECTION_MAX_ATTR  48
#define REID_MAX_VECTOR_LEN 64

typedef struct model_description model_description_t;

typedef struct kp
{
    float x, y, conf;
} kp_t;

typedef struct detection detection_t;

struct detection
{
    // main detection box. co-ords are normalized (0,0)-(1,1)
    // with respect to the original image not the detection ROI
    float x0,y0,x1,y1;
    float conf;
    // class index
    unsigned short cl;
    // original anchor box index (debug)
    unsigned short index;
    // overlap mask is a 64 bit mask of which "subblocks"
    // the detection box overlaps, when considering the whole
    // (0,0)->(1,1) 2D space as an 8x8 grid
    uint64_t overlap_mask;
    // 'subbox' is an fused 2nd detection
    // the intended main use case for this is to fuse faces
    // into person detections. But in future it could be
    // other things e.g. registration plate into vehicle
    // subbox_conf=0 => not present
    float subbox_x0,subbox_y0,subbox_x1,subbox_y1;
    float subbox_conf;
    // optional data items
    uint8_t num_face_points, num_pose_points, num_attr, reid_vector_len;
    // face points are typically Retinaface order
    // *note this is annoyingly L-R swapped vs pose points!!*
    // 0-R-Eye,1-L-Eye,2-Nose,3-R-Mouth,4-L-Mouth
    kp_t face_points[5];
    // pose points are typically MS COCO Pose order
    // 0-Nose, 1-L-Eye, 2-R-Eye, 3-Ear,...,15-L-Ankle,16-R-Ankle
    kp_t pose_points[17];
    // attributes float between 0 and 1 for each, e.g. 'has glasses'
    float attr[DETECTION_MAX_ATTR];
    // opaque vector for REID, can compare with cosine similarity
    // to check how 'similar' people look
    float reid[REID_MAX_VECTOR_LEN];
};

typedef struct detections
{
    // model description is defined in infer.h
    // this lets you get the class names list, attribute list, etc
    model_description_t *md;
    int num_detections;
    int max_detections;
    int num_person_detections;
    int num_face_detections;
    detection_t *person_dets; // points into the 'det' array above
    detection_t *face_dets;   // points into the 'det' array above
    // must be last!
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
void detections_unmap_roi(detections_t *dets, roi_t roi);
detections_t *detections_join(detections_t *dets1, detections_t *dets2);
image_t *draw_detections(detections_t *dets, image_t *img);
void show_detections(detections_t *dets);
void detections_generate_overlap_masks(detections_t *dets);
void fuse_face_person(detections_t *dets);

void match_detections_greedy(
    const detection_t *dets_a,
    int                num_dets_a,
    const detection_t *dets_b,
    int                num_dets_b,
    float            (*cost_fn)(const detection_t *, const detection_t *, void *),
    void              *ctx,
    uint8_t           *out_a_idx,
    uint8_t           *out_b_idx,
    int               *pOutCount
);

#endif