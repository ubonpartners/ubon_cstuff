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

void detection_destroy_keypoints(detection_t *d)
{
    if (d->kp)
    {
        free(d->kp);
        d->kp=0;
    }
}

void detection_create_keypoints(detection_t *d)
{
    if (!d) return;
    if (!d->kp)
    {
        d->kp=(keypoints_t *)malloc(sizeof(keypoints_t));
    }
}

void destroy_detections(detections_t *detections)
{
    if (!detections) return;
    for(int i=0;i<detections->num_detections;i++)
    {
        if (detections->det[i].kp!=0) detection_destroy_keypoints(&detections->det[i]);
    }
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

void detections_scale(detections_t *dets, float sx, float sy)
{
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=&dets->det[i];
        det->x0*=sx;
        det->y0*=sy;
        det->x1*=sx;
        det->y1*=sy;
        keypoints_t *kp=det->kp;
        if (kp)
        {
            for(int i=0;i<5;i++)
            {
                det->kp->face_points[i].x*=sx;
                det->kp->face_points[i].y*=sy;
            }
            for(int i=0;i<17;i++)
            {
                det->kp->pose_points[i].x*=sx;
                det->kp->pose_points[i].y*=sy;
            }
        }
    }
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

static void draw_kp_line(image_t *img, keypoints_t *kp, int a, int b)
{
    float thr=0.2;
    if ((kp->pose_points[a].conf<thr) || (kp->pose_points[b].conf<thr)) return;
    draw_line(img, kp->pose_points[a].x, kp->pose_points[a].y, kp->pose_points[b].x, kp->pose_points[b].y, 0x0000ff);
}

static void draw_kp_line(image_t *img, keypoints_t *kp, int a, int b, int c)
{
    float thr=0.2;
    if ((kp->pose_points[a].conf<thr) || (kp->pose_points[b].conf<thr) || (kp->pose_points[c].conf<thr)) return;
    draw_line(img, kp->pose_points[a].x, kp->pose_points[a].y, 
                   0.5*(kp->pose_points[b].x+kp->pose_points[c].x), 0.5*(kp->pose_points[b].y+kp->pose_points[c].y), 0x0000ff);
}

void show_detections(detections_t *dets)
{
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=&dets->det[i];
        printf("det %2d cls %8s conf %0.3f box[%0.3f,%0.3f,%0.3f,%0.3f]\n",i,
            (det->cl==0) ? "face" : "person", 
            det->conf, det->x0, det->y0, det->x1, det->y0);
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
        keypoints_t *kp=dets->det[i].kp;
        if (kp)
        {
            for(int j=0;j<5;j++)
            {
                if (kp->face_points[j].conf>0.5)
                {
                    draw_cross(x, kp->face_points[j].x, kp->face_points[j].y, 0.002, 0x00ff00);
                }
            }
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
                    detection_create_keypoints(d);
                    for(int i=0;i<5;i++)
                    {
                        d->kp->face_points[i].x=v[5+3*i+0];
                        d->kp->face_points[i].y=v[5+3*i+1];
                        d->kp->face_points[i].conf=v[5+3*i+2];
                    }
                }
                if ((n==56)||(n==71))
                {
                    // 5 + 15 + 51 = 71
                    int start=(n==56) ? 5 : 20;
                    detection_create_keypoints(d);
                    for(int i=0;i<17;i++)
                    {
                        d->kp->face_points[i].x=v[start+3*i+0];
                        d->kp->face_points[i].y=v[start+3*i+1];
                        d->kp->face_points[i].conf=v[start+3*i+2];
                    }
                }
            }
        }
    }
    fclose(f);
    return dets;
}
