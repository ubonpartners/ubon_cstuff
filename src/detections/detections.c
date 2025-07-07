#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mutex>
#include <float.h>  // for FLT_MIN
#include "assert.h"
#include "memory_stuff.h"
#include "detections.h"
#include "infer.h"
#include "match.h"
#include "log.h"
#include "misc.h"

static std::once_flag initFlag;
static allocation_tracker_t detection_allocation_tracker;
static block_allocator_t *detection_allocator;
static block_allocator_t *detection_list_allocator;

#define NORMAL_NUMBER_OF_DETECTIONS     200

static void detection_init()
{
    detection_allocator=block_allocator_create("detection", sizeof(detection_t));
    detection_list_allocator=block_allocator_create("detection list", sizeof(detection_list_t)+NORMAL_NUMBER_OF_DETECTIONS*sizeof(detection_t*));
}

detection_t *detection_create()
{
    detection_t *d=(detection_t*)block_alloc(detection_allocator);
    memset(d, 0, sizeof(detection_t));
    return d;
}

void detection_destroy(detection_t *d)
{
    if (block_reference_count(d)==1)
    {
        if (d->clip_embedding) embedding_destroy(d->clip_embedding);
        if (d->face_embedding) embedding_destroy(d->face_embedding);
    }
    block_free(d);
}

detection_t *detection_copy(detection_t *det)
{
    detection_t *det_new=detection_create();
    memcpy(det_new, det, sizeof(detection_t));
    return det_new;
}

detection_list_t *detection_list_copy(detection_list_t *dets)
{
    detection_list_t *dets_new=detection_list_create(dets->num_detections);
    dets_new->num_detections=dets->num_detections;
    for(int i=0;i<dets->num_detections;i++) dets_new->det[i]=detection_copy(dets->det[i]);
    dets_new->md=dets->md;
    return dets_new;
}

void detection_list_generate_overlap_masks(detection_list_t *dets)
{
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=dets->det[i];
        det->overlap_mask=box_to_8x8_mask(det->x0, det->y0, det->x1, det->y1);
    }
}

static void detection_list_free_callback(void *context, void *block)
{
    detection_list_t *dets=(detection_list_t *)block;
    for(int i=0;i<dets->num_detections;i++) detection_destroy(dets->det[i]);
}

detection_list_t *detection_list_create(int max_detections)
{
    std::call_once(initFlag, detection_init);
    int sz=sizeof(detection_list_t)+max_detections*sizeof(detection_t *);
    detection_list_t *dets=(detection_list_t *)block_alloc(detection_list_allocator, sz);
    if (!dets) return 0;
    memset(dets, 0, sz);
    dets->max_detections=max_detections;
    block_set_free_callback(dets, 0, detection_list_free_callback);
    return dets;
}

void detection_list_destroy(detection_list_t *detections)
{
    if (!detections) return;
    block_free(detections);
}

detection_t *detection_list_add_end(detection_list_t *detections)
{
    if (detections->num_detections>=detections->max_detections) return 0;
    assert(detections->det[detections->num_detections]==0);
    detection_t *det=detection_create();
    detections->det[detections->num_detections++]=det;
    return det;
}

void detection_list_append_copy(detection_list_t *detections, detection_t *det)
{
    detection_t *det1=detection_list_add_end(detections);
    if (det1) memcpy(det1, det, sizeof(detection_t));
}

static int compare_detections(const void *a, const void *b)
{
    float conf_a = (*((detection_t**)a))->conf;
    float conf_b = (*((detection_t**)b))->conf;
    if (conf_a < conf_b) return 1;
    if (conf_a > conf_b) return -1;
    return 0;
}

void detections_list_sort_descending_conf(detection_list_t *detections)
{
    qsort(detections->det, detections->num_detections, sizeof(detection_t*), compare_detections);
}

float det_iou(detection_t *deta, detection_t *detb)
{
    float iw=fmaxf(0, fminf(deta->x1, detb->x1)-fmaxf(deta->x0, detb->x0));
    float ih=fmaxf(0, fminf(deta->y1, detb->y1)-fmaxf(deta->y0, detb->y0));
    float ai=iw*ih;
    float aa=(deta->x1-deta->x0)*(deta->y1-deta->y0);
    float ab=(detb->x1-detb->x0)*(detb->y1-detb->y0);
    float iou=ai/(aa+ab-ai+1e-7);
    return iou;
}

void detection_list_nms_inplace(detection_list_t *dets, float iou_thr)
{
    detections_list_sort_descending_conf(dets);

    int num_detections=dets->num_detections;
    for(int i=0;i<num_detections;i++)
    {
        detection_t *det=dets->det[i];
        int new_num_detections=i+1;
        int cl=det->cl;
        for(int j=i+1;j<num_detections;j++)
        {
            detection_t *det1=dets->det[j];
            bool kill=(det1->cl==cl)&&(det_iou(det, det1)>iou_thr);
            if (!kill)
                dets->det[new_num_detections++]=det1;
            else
                detection_destroy(det1);
        }
        num_detections=new_num_detections;
    }
    dets->num_detections=num_detections;
}

const char *detection_list_get_classname(detection_list_t *dets, int cl)
{
    const char * classname=0;
    static const char *guess_classes[5]={"person?","face?","vehicle?","animal?","weapon?"};
    if (dets->md!=0) classname=dets->md->class_names[cl].c_str();
    if (classname==0 && cl<5) classname=guess_classes[cl];
    return classname;
}

static inline float clip_01(float x)
{
    return std::min(std::max(x, 0.0f), 1.0f);
}

void detection_list_scale_add(detection_list_t *dets, float sx, float sy, float dx, float dy)
{
    if (!dets) return;
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=dets->det[i];
        det->x0=clip_01(dx+det->x0*sx);
        det->y0=clip_01(dy+det->y0*sy);
        det->x1=clip_01(dx+det->x1*sx);
        det->y1=clip_01(dy+det->y1*sy);
        if (det->subbox_conf>0)
        {
            det->subbox_x0=clip_01(dx+det->subbox_x0*sx);
            det->subbox_y0=clip_01(dy+det->subbox_y0*sy);
            det->subbox_x1=clip_01(dx+det->subbox_x1*sx);
            det->subbox_y1=clip_01(dy+det->subbox_y1*sy);
        }
        for(int i=0;i<det->num_face_points;i++)
        {
            det->face_points[i].x=clip_01(dx+det->face_points[i].x*sx);
            det->face_points[i].y=clip_01(dy+det->face_points[i].y*sy);
        }
        for(int i=0;i<det->num_pose_points;i++)
        {
            det->pose_points[i].x=clip_01(dx+det->pose_points[i].x*sx);
            det->pose_points[i].y=clip_01(dy+det->pose_points[i].y*sy);
        }
    }
}

void detection_list_scale_add2(detection_list_t *dets, float sx, float sy, float dx, float dy)
{
    if (!dets) return;
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=dets->det[i];
        det->x0=clip_01((det->x0-dx)*sx);
        det->y0=clip_01((det->y0-dy)*sy);
        det->x1=clip_01((det->x1-dx)*sx);
        det->y1=clip_01((det->y1-dy)*sy);
        if (det->subbox_conf>0)
        {
            det->subbox_x0=clip_01((det->subbox_x0-dx)*sx);
            det->subbox_y0=clip_01((det->subbox_y0-dy)*sy);
            det->subbox_x1=clip_01((det->subbox_x1-dx)*sx);
            det->subbox_y1=clip_01((det->subbox_y1-dy)*sy);
        }
        for(int i=0;i<det->num_face_points;i++)
        {
            det->face_points[i].x=clip_01((det->face_points[i].x-dx)*sx);
            det->face_points[i].y=clip_01((det->face_points[i].y-dy)*sy);
        }
        for(int i=0;i<det->num_pose_points;i++)
        {
            det->pose_points[i].x=clip_01((det->pose_points[i].x-dx)*sx);
            det->pose_points[i].y=clip_01((det->pose_points[i].y-dy)*sy);
        }
    }
}

void detection_list_scale(detection_list_t *dets, float sx, float sy)
{
    detection_list_scale_add(dets, sx, sy, 0, 0);
}

void detection_list_unmap_roi(detection_list_t *dets, roi_t roi)
{
    float sx=roi.box[2]-roi.box[0];
    float sy=roi.box[3]-roi.box[1];
    float dx=roi.box[0];
    float dy=roi.box[1];
    detection_list_scale_add(dets, sx, sy, dx, dy);
}

detection_list_t *detection_list_join(detection_list_t *dets1, detection_list_t *dets2)
{
    detection_list_t *ret=detection_list_create(dets1->num_detections+dets2->num_detections);
    for(int i=0;i<dets1->num_detections;i++) detection_list_append_copy(ret, dets1->det[i]);
    for(int i=0;i<dets2->num_detections;i++) detection_list_append_copy(ret, dets2->det[i]);
    return ret;
}

float detection_face_quality_score(detection_t *det) {
    kp_t *face_points=det->face_points;
    if (det->num_face_points!=5) return -1;
    // --- 1) Confidence aggregates ---
    float conf_eye  = 0.5f * (face_points[0].conf + face_points[1].conf);
    float conf_nose = face_points[2].conf;
    float conf_mouth = 0.5f * (face_points[3].conf + face_points[4].conf);
    // overall mean conf
    float conf_overall = (conf_eye + conf_nose + conf_mouth) / 3.0f;
    if (conf_overall<0.1) return 0.0f;

    // --- 2) Size factor (inter‐ocular distance) ---
    float dx = face_points[1].x - face_points[0].x;
    float dy = face_points[1].y - face_points[0].y;
    float iod = sqrtf(dx*dx + dy*dy);              // normalized [0,√2]
    float size_score = fminf(1.0f, iod * 1.5f);    // tune 1.5→ desired scale
    size_score *= conf_eye;                        // down‐weight if eyes are uncertain
    // --- 3) Roll (eye‐line tilt) ---
    float roll = fabsf(atan2f(dy, dx));            // radians
    const float MAX_ROLL = 30.0f * (M_PI/180.0f);   // 30°
    float roll_score = (roll >= MAX_ROLL)
                     ? 0.1f
                     : 1.0f - (roll / MAX_ROLL);
    roll_score *= conf_eye;                        // again weight by eye conf

    // --- 4) Yaw proxy (nose offset) ---
    float ex = 0.5f * (face_points[0].x + face_points[1].x);
    float nx = face_points[2].x;
    float offset = fabsf(nx - ex);
    float offset_ratio = (iod > 0.0f) ? (offset / iod) : 1.0f;
    float yaw_score = (offset_ratio >= 1.0f)
                    ? 0.1f
                    : 1.0f - offset_ratio;
    // weight by the weaker of (eyes, nose)
    float conf_yaw = fminf(conf_eye, conf_nose);
    yaw_score *= conf_yaw;

    // --- 5) Combine components and overall conf ---
    float combined = size_score * roll_score * yaw_score;
    float final_score = combined * conf_overall;

    // clamp
    if (final_score < 0.0f) final_score = 0.0f;
    if (final_score > 1.0f) final_score = 1.0f;
    return final_score;
}
