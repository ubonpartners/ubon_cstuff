#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "assert.h"
#include "detections.h"
#include "match.h"
#include "log.h"
#include "infer.h"

static float person_face_match_cost(const detection_t *person, const detection_t *face, void *context)
{
    float px0 = person->x0, py0 = person->y0, px1 = person->x1, py1 = person->y1;
    float fx0 = face->x0,   fy0 = face->y0,   fx1 = face->x1,   fy1 = face->y1;

    float ix0 = fmaxf(px0, fx0);
    float iy0 = fmaxf(py0, fy0);
    float ix1 = fminf(px1, fx1);
    float iy1 = fminf(py1, fy1);

    float iw = fmaxf(0.0f, ix1 - ix0);
    float ih = fmaxf(0.0f, iy1 - iy0);
    if (iw==0) return 0;
    if (ih==0) return 0;

    float inter_area = iw * ih;

    float area_person = fmaxf((px1 - px0) * (py1 - py0), FLT_MIN);
    float area_face   = fmaxf((fx1 - fx0) * (fy1 - fy0), FLT_MIN);
    float min_area    = fminf(area_person, area_face);

    float ioma=inter_area / min_area;
    float score=ioma;

    // prefer face not too small compared to body
    if ((px1-px0)<10*(fx1-fx0)) score+=0.2;
    // prefer face is not bigger than body
    if ((px1-px0)>=(fx1-fx0)) score+=0.2;
    // perfer top of face top half of body
    if (fy1<(0.5*(py0+py1))) score+=0.2;
    // todo : add score for face keypoints matching
    return ioma;
}

int match_detections_greedy(
    detection_t **dets_a,
    int                num_dets_a,
    detection_t **dets_b,
    int                num_dets_b,
    float            (*cost_fn)(const detection_t *, const detection_t *, void *),
    void              *ctx,
    uint16_t           *out_a_idx,
    uint16_t           *out_b_idx
)
{
    if (num_dets_a <= 0 || num_dets_b <= 0) {
        return 0;
    }

    /* 1) Extract overlap_mask arrays */
    uint64_t maskA[num_dets_a], maskB[num_dets_b];
    void *dets_a_ptr[num_dets_a];
    void *dets_b_ptr[num_dets_b];

    for (int i = 0; i < num_dets_a; ++i) {
        maskA[i] = dets_a[i]->overlap_mask;
        dets_a_ptr[i]=(void*)dets_a[i];
    }
    for (int j = 0; j < num_dets_b; ++j) {
        maskB[j] = dets_b[j]->overlap_mask;
        dets_b_ptr[j]=(void*)dets_b[j];
    }

    return match_greedy(
        (const void **)dets_a_ptr, maskA, num_dets_a,
        (const void **)dets_b_ptr, maskB, num_dets_b,
        (float (*)(const void*, const void*, void*))cost_fn, ctx,
        out_a_idx, out_b_idx);
}

static float score_box_iou(const detection_t *da, const detection_t *db, void *ctx)
{
    float x_left = fmaxf(da->x0, db->x0);
    float y_top = fmaxf(da->y0, db->y0);
    float x_right = fminf(da->x1, db->x1);
    float y_bottom = fminf(da->y1, db->y1);

    if (x_right < x_left || y_bottom < y_top)
        return 0.0f;

    float inter=(x_right - x_left) * (y_bottom - y_top);

    float area_a = (da->x1 - da->x0) * (da->y1 - da->y0);
    float area_b = (db->x1 - db->x0) * (db->y1 - db->y0);

    float union_area = area_a + area_b - inter;

    if (union_area <= 0.0f)
        return 0.0f;

    float thr=*((float *)ctx);

    float ret=inter / union_area;
    if (ret<thr) return 0.0;
    return ret;
}

static float kp_iou(const kp_t *a, const kp_t *b, const float *scales, int n, float area)
{
    float ss=area*0.53;
    float num=0;
    float denom=0;
    for(int i=0;i<n;i++)
    {
        if (b[i].conf>0.3) // if GT is labelled
        {
            float dx=a[i].x-b[i].x;
            float dy=a[i].y-b[i].y;
            num+=expf(-(dx*dx+dy*dy)/(2.0*ss*scales[i]*scales[i]*4+1e-7));
            denom+=1.0;
        }
    }
    float iou=num/(denom+1e-7);
    return iou;
}

static float score_facepoint_iou(const detection_t *da, const detection_t *db, void *ctx)
{
    assert(da->num_face_points==5);
    assert(db->num_face_points==5);
    assert(da->cl==db->cl);
    if (score_box_iou(da, db, ctx)<=0) return 0;
    const float face_scales[5]={0.025, 0.025, 0.026, 0.025, 0.025};
    float box_a=(db->x1-db->x0)*(db->y1-db->y0);
    float iou=kp_iou(da->face_points, db->face_points, face_scales, 5, box_a);
    float thr=*((float *)ctx);
    if (iou<thr) return 0.0;
    return iou;
}

static float score_posepoint_iou(const detection_t *da, const detection_t *db, void *ctx)
{
    assert(da->num_pose_points==17);
    assert(db->num_pose_points==17);
    assert(da->cl==db->cl);
    if (score_box_iou(da, db, ctx)<=0) return 0;

    const float pose_scales[17]={0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089};
    float box_a=(db->x1-db->x0)*(db->y1-db->y0);

    float iou=kp_iou(da->pose_points, db->pose_points, pose_scales, 17, box_a);
    float thr=*((float *)ctx);
    if (iou<thr) return 0.0;
    return iou;
}

int match_box_iou(
    detection_t **dets_a,
    int                num_dets_a,
    detection_t **dets_b,
    int                num_dets_b,
    uint16_t           *out_a_idx,
    uint16_t           *out_b_idx,
    float              iou_thr,
    match_type_t      match_type
)
{
    assert(num_dets_a<=MAX_DETS);
    assert(num_dets_b<=MAX_DETS);
    for(int i=0;i<num_dets_a;i++)
    {
        detection_t *det=dets_a[i];
        det->overlap_mask=box_to_8x8_mask(det->x0, det->y0, det->x1, det->y1);
    }
    for(int i=0;i<num_dets_b;i++)
    {
        detection_t *det=dets_b[i];
        det->overlap_mask=box_to_8x8_mask(det->x0, det->y0, det->x1, det->y1);
    }

    int ret=0;
    if (match_type==MATCH_TYPE_BOX_IOU)
        ret=match_detections_greedy(dets_a, num_dets_a,dets_b, num_dets_b,score_box_iou, &iou_thr,out_a_idx, out_b_idx);
    else if (match_type==MATCH_TYPE_FACE_KP)
        ret=match_detections_greedy(dets_a, num_dets_a,dets_b, num_dets_b,score_facepoint_iou, &iou_thr,out_a_idx, out_b_idx);
    else if (match_type==MATCH_TYPE_POSE_KP)
        ret=match_detections_greedy(dets_a, num_dets_a,dets_b, num_dets_b,score_posepoint_iou, &iou_thr,out_a_idx, out_b_idx);
    else
    {
        assert(0);
    }
    return ret;
}

void detection_list_fuse_face_person(detection_list_t *dets)
{
    int num=0;
    uint16_t person_idx[dets->num_person_detections];
    uint16_t face_idx[dets->num_face_detections];

    //printf("nperson %d nface %d\n",dets->num_person_detections, dets->num_face_detections);

    num=match_detections_greedy(dets->person_dets, dets->num_person_detections,
                            dets->face_dets, dets->num_face_detections,
                            person_face_match_cost, 0,
                            person_idx, face_idx);


    for (int i=0;i<num;i++)
    {
        int pi=person_idx[i];
        int fi=face_idx[i];
        detection_t *person=dets->person_dets[pi];
        detection_t *face=dets->face_dets[fi];
        person->subbox_x0=face->x0;
        person->subbox_y0=face->y0;
        person->subbox_x1=face->x1;
        person->subbox_y1=face->y1;
        person->subbox_conf=face->conf;
        person->num_face_points=face->num_face_points;
        for(int j=0;j<face->num_face_points;j++) person->face_points[j]=face->face_points[j];
        //printf("index %2d match %d with %d\n",i,person_idx[i],face_idx[i]);
    }
    int old_num=dets->num_detections;
    int new_num=0;
    int fc=dets->md->face_class_index;

    // delete face detections from list (including unmatched ones)
    for (int i=0;i<old_num;i++)
    {
        if (dets->det[i]->cl==fc)
            detection_destroy(dets->det[i]);
        else
            dets->det[new_num++]=dets->det[i];
    }
    dets->num_detections=new_num;
    dets->face_dets=0;
    dets->num_face_detections=0;
}
