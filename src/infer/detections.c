#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>  // for FLT_MIN
#include "assert.h"
#include "detections.h"
#include "image.h"
#include "infer.h"
#include "match.h"

void detections_generate_overlap_masks(detections_t *dets)
{
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=dets->det+i;
        det->overlap_mask=box_to_8x8_mask(det->x0, det->y0, det->x1, det->y1);
    }
}

detections_t *create_detections(int max_detections)
{
    int sz=sizeof(detections_t)+max_detections*sizeof(detection_t);
    detections_t *dets=(detections_t *)malloc(sz);
    if (!dets) return 0;
    memset(dets, 0, sz);
    dets->max_detections=max_detections;
    return dets;
}

void destroy_detections(detections_t *detections)
{
    if (!detections) return;
    free(detections);
}

detection_t *detection_add_end(detections_t *detections)
{
    if (detections->num_detections>=detections->max_detections) return 0;
    detection_t *det=&detections->det[detections->num_detections];
    detections->num_detections++;
    memset(det, 0, sizeof(detection_t));
    return det;
}

static int compare_detections(const void *a, const void *b)
{
    float conf_a = ((detection_t*)a)->conf;
    float conf_b = ((detection_t*)b)->conf;
    if (conf_a < conf_b) return 1;
    if (conf_a > conf_b) return -1;
    return 0;
}

void detections_sort_descending_conf(detections_t *detections)
{
    qsort(detections->det, detections->num_detections, sizeof(detection_t), compare_detections);
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

void detections_nms_inplace(detections_t *dets, float iou_thr)
{
    detections_sort_descending_conf(dets);
    // NMS - set conf to 0 for 'suppressed' detections
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=&dets->det[i];
        for(int j=i+1;j<dets->num_detections;j++)
        {
            detection_t *det1=&dets->det[j];
            if ((det1->cl==det->cl)&&(det1->conf!=0))
            {
                if (det_iou(det, det1)>iou_thr) det1->conf=0;
            }
        }
    }
    // delete detections where we set the confidence to 0
    int num_out=0;
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=&dets->det[i];
        if (det->conf>0)
        {
            if (i!=num_out) memcpy(&dets->det[num_out], det, sizeof(detection_t));
            num_out++;
        }
    }
    dets->num_detections=num_out;
}

static void draw_line(image_t *img, float x0, float y0, float x1, float y1, int clr)
{
    int steps=(int)(fmaxf(fabsf(x1-x0)*img->width, fabsf(y1-y0)*img->height)+0.999);
    if (steps==0) return;
    assert(img->format==IMAGE_FORMAT_RGB24_HOST);
    assert(img->rgb!=0);
    for(int i=0;i<=steps;i++)
    {
        float l=(i+0.0)/(steps+0.0);
        float x=x0*(1.0-l)+x1*l;
        float y=y0*(1.0-l)+y1*l;
        int xi=(int)(x*img->width);
        int yi=(int)(y*img->height);
        if ((xi<0)||(yi<0)||(xi>=img->width)||(yi>=img->height)) continue;
        img->rgb[xi*3+yi*img->stride_rgb+0]=(clr>>16)&0xff;
        img->rgb[xi*3+yi*img->stride_rgb+1]=(clr>>8)&0xff;
        img->rgb[xi*3+yi*img->stride_rgb+2]=(clr>>0)&0xff;
    }
}

static void draw_detection(detection_t *d, image_t *img)
{
    int clr=0xff0000;
    assert(img->format==IMAGE_FORMAT_RGB24_HOST);
    if (d->cl==0) clr=0xffff00;
    draw_line(img, d->x0, d->y0, d->x1, d->y0, clr);
    draw_line(img, d->x0, d->y1, d->x1, d->y1, clr);
    draw_line(img, d->x0, d->y0, d->x0, d->y1, clr);
    draw_line(img, d->x1, d->y0, d->x1, d->y1, clr);
}

static void draw_cross(image_t *img, float cx, float cy, float w, int clr)
{
    draw_line(img, cx-w, cy-w, cx+w, cy+w, clr);
    draw_line(img, cx-w, cy+w, cx+w, cy-w, clr);
}

static void draw_kp_line(image_t *img, kp_t *kp, int a, int b)
{
    float thr=0.2;
    if ((kp[a].conf<thr) || (kp[b].conf<thr)) return;
    draw_line(img, kp[a].x, kp[a].y, kp[b].x, kp[b].y, 0x0000ff);
}

static void draw_kp_line(image_t *img, kp_t *kp, int a, int b, int c)
{
    float thr=0.2;
    if ((kp[a].conf<thr) || (kp[b].conf<thr) || (kp[c].conf<thr)) return;
    draw_line(img, kp[a].x, kp[a].y,
                   0.5*(kp[b].x+kp[c].x), 0.5*(kp[b].y+kp[c].y), 0x0000ff);
}

void show_detections(detections_t *dets)
{
    const char *guess_classes[5]={"person?","face?","vehicle?","animal?","weapon?"};
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=&dets->det[i];
        const char * classname=0;
        if (dets->md!=0) classname=dets->md->class_names[det->cl].c_str();
        if (classname==0 && det->cl<5) classname=guess_classes[det->cl];
        printf("det %2d cls %8s conf %0.3f idx:%3d box[%0.3f,%0.3f,%0.3f,%0.3f] area %0.3f\n",i,
            classname,
            det->conf, det->index, det->x0, det->y0, det->x1, det->y0,
            (det->y1-det->y0)*(det->x1-det->x0));
    }
}

image_t *draw_detections(detections_t *dets, image_t *img)
{
    image_sync(img);
    image_t *x=image_convert(img, IMAGE_FORMAT_RGB24_HOST);
    assert(x!=0);
    assert(x->format==IMAGE_FORMAT_RGB24_HOST);
    for(int i=0;i<dets->num_detections;i++)
    {
        draw_detection(&dets->det[i], x);

        for(int j=0;j<dets->det[i].num_face_points;j++)
        {
            if (dets->det[i].face_points[j].conf>0.5)
            {
                draw_cross(x, dets->det[i].face_points[j].x, dets->det[i].face_points[j].y, 0.002, 0x00ff00);
            }
        }

        kp_t *kp=dets->det[i].pose_points;
        draw_kp_line(x, kp, 0, 1);
        draw_kp_line(x, kp, 0, 2);
        draw_kp_line(x, kp, 0, 5, 6);

        draw_kp_line(x, kp, 1, 3);
        draw_kp_line(x, kp, 2, 4);

        draw_kp_line(x, kp, 5, 6);
        draw_kp_line(x, kp, 5, 11);
        draw_kp_line(x, kp, 6, 12);
        draw_kp_line(x, kp, 11, 12);

        draw_kp_line(x, kp, 5, 7);
        draw_kp_line(x, kp, 7, 9);

        draw_kp_line(x, kp, 6, 8);
        draw_kp_line(x, kp, 8, 10);

        draw_kp_line(x, kp, 11, 13);
        draw_kp_line(x, kp, 13, 15);

        draw_kp_line(x, kp, 12, 14);
        draw_kp_line(x, kp, 14, 16);

    }
    return x;
}

detections_t *load_detections(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f) return 0;

    char buffer[1024];
    detections_t *dets=create_detections(300);
    while(fgets(buffer, sizeof(buffer), f) != NULL)
    {
        float v[128];
        int n=0;
        const char *ptr = buffer;
        while(n<128)
        {
            if ((*ptr=='\0') || (*ptr=='\n')) break;
            if (1!=sscanf(ptr, "%f", &v[n])) break;
            n++;
            while (*ptr != ' ' && *ptr != '\0' && *ptr != '\n') ptr++;
            while (*ptr == ' ') ptr++;
        }
        if (n>=5)
        {
            detection_t *d=detection_add_end(dets);
            if (d)
            {
                d->x0=v[1]-0.5*v[3];
                d->x1=v[1]+0.5*v[3];
                d->y0=v[2]-0.5*v[4];
                d->y1=v[2]+0.5*v[4];
                d->conf=1.0;
                d->cl=(int)v[0];
                if ((n==20)||(n==71))
                {
                    d->num_face_points=5;
                    for(int i=0;i<5;i++)
                    {
                        d->face_points[i].x=v[5+3*i+0];
                        d->face_points[i].y=v[5+3*i+1];
                        d->face_points[i].conf=v[5+3*i+2];
                    }
                }
                if ((n==56)||(n==71))
                {
                    // 5 + 15 + 51 = 71
                    int start=(n==56) ? 5 : 20;
                    d->num_face_points=5;
                    for(int i=0;i<17;i++)
                    {
                        d->face_points[i].x=v[start+3*i+0];
                        d->face_points[i].y=v[start+3*i+1];
                        d->face_points[i].conf=v[start+3*i+2];
                    }
                }
            }
        }
    }
    fclose(f);
    return dets;
}

static inline float clip_01(float x)
{
    return std::min(std::max(x, 0.0f), 1.0f);
}

void detections_scale_add(detections_t *dets, float sx, float sy, float dx, float dy)
{
    if (!dets) return;
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=&dets->det[i];
        det->x0=clip_01(dx+det->x0*sx);
        det->y0=clip_01(dy+det->y0*sy);
        det->x1=clip_01(dx+det->x1*sx);
        det->y1=clip_01(dy+det->y1*sy);
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

void detections_scale_add2(detections_t *dets, float sx, float sy, float dx, float dy)
{
    if (!dets) return;
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=&dets->det[i];
        det->x0=clip_01((det->x0-dx)*sx);
        det->y0=clip_01((det->y0-dy)*sy);
        det->x1=clip_01((det->x1-dx)*sx);
        det->y1=clip_01((det->y1-dy)*sy);
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

void detections_scale(detections_t *dets, float sx, float sy)
{
    detections_scale_add(dets, sx, sy, 0, 0);
}

void detections_unmap_roi(detections_t *dets, roi_t roi)
{
    float sx=roi.box[2]-roi.box[0];
    float sy=roi.box[3]-roi.box[1];
    float dx=roi.box[0];
    float dy=roi.box[1];
    detections_scale_add(dets, sx, sy, dx, dy);
}

detections_t *detections_join(detections_t *dets1, detections_t *dets2)
{
    detections_t *ret=create_detections(dets1->num_detections+dets2->num_detections);
    memcpy(ret->det, dets1->det, sizeof(detection_t)*dets1->num_detections);
    memcpy(ret->det+dets1->num_detections, dets2->det, sizeof(detection_t)*dets2->num_detections);
    ret->num_detections=dets1->num_detections+dets2->num_detections;
    return ret;
}


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

void fuse_face_person(detections_t *dets)
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
        detection_t *person=&dets->person_dets[pi];
        detection_t *face=&dets->face_dets[fi];
        person->subbox_x0=face->x0;
        person->subbox_y0=face->y0;
        person->subbox_x1=face->x1;
        person->subbox_y1=face->y1;
        person->subbox_conf=face->conf;
        person->num_face_points=face->num_face_points;
        for(int j=0;j<face->num_face_points;j++) person->face_points[j]=face->face_points[j];
        //printf("index %2d match %d with %d\n",i,person_idx[i],face_idx[i]);
    }
    /*if (num==dets->num_face_detections)
    {
        printf("ALL faces matched\n");
    }
    else
    {
        printf("%d unmatched faces\n", dets->num_face_detections-num);
    }*/
    dets->num_face_detections=0;
}

int match_detections_greedy(
    const detection_t *dets_a,
    int                num_dets_a,
    const detection_t *dets_b,
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
    uint64_t maskA[MAX_DETS], maskB[MAX_DETS];
    void *dets_a_ptr[MAX_DETS];
    void *dets_b_ptr[MAX_DETS];

    for (int i = 0; i < num_dets_a; ++i) {
        maskA[i] = dets_a[i].overlap_mask;
        dets_a_ptr[i]=(void*)&dets_a[i];
    }
    for (int j = 0; j < num_dets_b; ++j) {
        maskB[j] = dets_b[j].overlap_mask;
        dets_b_ptr[j]=(void*)&dets_b[j];
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
    const detection_t *dets_a,
    int                num_dets_a,
    const detection_t *dets_b,
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
        detection_t *det=(detection_t *)&dets_a[i];
        det->overlap_mask=box_to_8x8_mask(det->x0, det->y0, det->x1, det->y1);
    }
    for(int i=0;i<num_dets_b;i++)
    {
        detection_t *det=(detection_t *)&dets_b[i];
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