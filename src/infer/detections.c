#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "assert.h"
#include "detections.h"
#include "image.h"

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
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=&dets->det[i];
        printf("det %2d cls %8s conf %0.3f idx:%3d box[%0.3f,%0.3f,%0.3f,%0.3f] area %0.3f\n",i,
            (det->cl==0) ? "face" : "person",
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

static void detections_scale_add(detections_t *dets, float sx, float sy, float dx, float dy)
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
