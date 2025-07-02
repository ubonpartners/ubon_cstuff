#ifndef __DETECTIONS_H
#define __DETECTIONS_H

#include <stdint.h>

#define DETECTION_MAX_ATTR  48
#define REID_MAX_VECTOR_LEN 64
#define MAX_DETS            65535 // per class

typedef struct model_description model_description_t;

typedef struct kp
{
    float x, y, conf;
} kp_t;

typedef struct detection detection_t;

struct detection
{
    uint32_t marker;
    // main detection box. co-ords are normalized (0,0)-(1,1)
    // with respect to the original image not the detection ROI
    float x0,y0,x1,y1;
    float conf;
    // class index
    unsigned short cl;
    // original anchor box index (debug)
    unsigned short index;
    // track ID
    uint64_t track_id;
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

typedef struct detection_list
{
    // model description is defined in infer.h
    // this lets you get the class names list, attribute list, etc
    model_description_t *md;
    int num_detections;
    int max_detections;
    int num_person_detections;
    int num_face_detections;
    detection_t **person_dets; // points into the 'det' array above
    detection_t **face_dets;   // points into the 'det' array above
    // must be last!
    detection_t *det[1];
} detection_list_t;

#include "image.h"

detection_t *detection_create();
void detection_destroy(detection_t *det);
detection_t *detection_copy(detection_t *det);

detection_list_t *detection_list_copy(detection_list_t *dets);
detection_list_t *detection_list_create(int max_detections);
detection_list_t *detection_list_load(const char *filename);
void detection_list_destroy(detection_list_t *detections);
detection_t *detection_list_add_end(detection_list_t *detections);
void detection_list_append_copy(detection_list_t *detections, detection_t *det);
void detection_list_nms_inplace(detection_list_t *detections, float iou_thr);
void detections_list_sort_descending_conf(detection_list_t *detections);
void detection_list_scale(detection_list_t *dets, float sx, float sy);
void detection_list_scale_add(detection_list_t *dets, float sx, float sy, float dx, float dy);
void detection_list_scale_add2(detection_list_t *dets, float sx, float sy, float dx, float dy);
void detection_list_unmap_roi(detection_list_t *dets, roi_t roi);
detection_list_t *detection_list_join(detection_list_t *dets1, detection_list_t *dets2);
image_t *detection_list_draw(detection_list_t *dets, image_t *img);
void detection_list_show(detection_list_t *dets, bool log_only=false);
void detection_list_generate_overlap_masks(detection_list_t *dets);
void detection_list_fuse_face_person(detection_list_t *dets);
const char *detection_list_get_classname(detection_list_t *dets, int cl);

float detection_box_iou(const detection_t *da, const detection_t *db);
int match_detections_greedy(
    detection_t        **dets_a,
    int                num_dets_a,
    detection_t        **dets_b,
    int                num_dets_b,
    float            (*cost_fn)(const detection_t *, const detection_t *, void *),
    void              *ctx,
    uint16_t           *out_a_idx,
    uint16_t           *out_b_idx,
    float              *out_score=0,
    bool               do_debug=false
);

typedef enum match_type
{
    MATCH_TYPE_BOX_IOU=0,
    MATCH_TYPE_FACE_KP=1,
    MATCH_TYPE_POSE_KP=2
} match_type_t;

int match_box_iou(
    detection_t        **dets_a,
    int                num_dets_a,
    detection_t        **dets_b,
    int                num_dets_b,
    uint16_t           *out_a_idx,
    uint16_t           *out_b_idx,
    float             iou_thr,
    match_type_t      match_type
);
#endif