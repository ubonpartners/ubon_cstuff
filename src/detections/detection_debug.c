#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mutex>
#include <float.h>  // for FLT_MIN
#include "assert.h"
#include "memory_stuff.h"
#include "detections.h"
#include "image.h"
#include "image_draw.h"
#include "infer.h"
#include "match.h"
#include "log.h"
#include "misc.h"
#include "fiqa.h"

detection_list_t *detection_list_load(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f) return 0;

    char buffer[1024];
    detection_list_t *dets=detection_list_create(300);
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
            detection_t *d=detection_list_add_end(dets);
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
                    d->num_pose_points=17;
                    for(int i=0;i<17;i++)
                    {
                        d->pose_points[i].x=v[start+3*i+0];
                        d->pose_points[i].y=v[start+3*i+1];
                        d->pose_points[i].conf=v[start+3*i+2];
                    }
                }
            }
        }
    }
    fclose(f);
    return dets;
}

static void draw_cross(image_t *img, float cx, float cy, float w, uint32_t argb)
{
    image_draw_line(img, cx-w, cy-w, cx+w, cy+w, argb);
    image_draw_line(img, cx-w, cy+w, cx+w, cy-w, argb);
}

static void draw_kp_line(image_t *img, kp_t *kp, int a, int b)
{
    float thr=0.2;
    if ((kp[a].conf<thr) || (kp[b].conf<thr)) return;
    image_draw_line(img, kp[a].x, kp[a].y, kp[b].x, kp[b].y, 0xff0000ff);
}

static void draw_kp_line(image_t *img, kp_t *kp, int a, int b, int c)
{
    float thr=0.2;
    if ((kp[a].conf<thr) || (kp[b].conf<thr) || (kp[c].conf<thr)) return;
    image_draw_line(img, kp[a].x, kp[a].y,
                   0.5*(kp[b].x+kp[c].x), 0.5*(kp[b].y+kp[c].y), 0xff0000ff);
}

image_t *detection_list_draw(detection_list_t *dets, image_t *img)
{
    if (!img) return 0;
    image_t *x=image_convert(img, IMAGE_FORMAT_RGB24_HOST);
    image_sync(x);
    assert(x!=0);
    assert(x->format==IMAGE_FORMAT_RGB24_HOST);
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=dets->det[i];
        if (det->cl!=0) continue;

        image_draw_box(x, det->x0, det->y0, det->x1, det->y1, 0xffff0000);
        if (det->subbox_conf>0)
        {
            image_draw_box(x, det->subbox_x0, det->subbox_y0,det->subbox_x1, det->subbox_y1, 0xff00ff00);
        }
        for(int j=0;j<det->num_face_points;j++)
        {
            if (det->face_points[j].conf>0.05)
            {
                uint32_t a=255*det->face_points[j].conf;
                uint32_t clr=0xff00+(a<<24);
                draw_cross(x, det->face_points[j].x, det->face_points[j].y, 0.002, clr);
            }
        }

        kp_t *kp=det->pose_points;
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
        const char * classname=detection_list_get_classname(dets, det->cl);
        char text[256];
        float offs=0;
        snprintf(text, 255, "ID:%lx %s",det->track_id,classname);
        image_draw_text(x, det->x0, det->y0+offs, text, 0xffffffff);
        offs+=0.02;

        //snprintf(text, 255, "FQ:%0.4f",detection_face_quality_score(det));
        //image_draw_text(x, det->x0, det->y0+offs, text, 0xffffffff);
        //offs+=0.02;

        if (1)
        {
            float bf=cevo_bestface_score(det);
            snprintf(text, 255, "BF:%0.3f",bf);
            image_draw_text(x, det->x0, det->y0+offs, text, 0xffffffff);
            offs+=0.02;
        }
        if (det->fiqa_embedding!=0)
        {
            snprintf(text, 255, "FIQA:%0.3f",fiqa_embedding_quality(det->fiqa_embedding));
            image_draw_text(x, det->x0, det->y0+offs, text, 0xffffffff);
            offs+=0.02;
        }
    }
    return x;
}

void detection_list_show(detection_list_t *dets, bool log_only)
{
    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=dets->det[i];
        const char * classname=detection_list_get_classname(dets, det->cl);
        if (log_only==false)
        {
            printf("DLS det %2d cls %8s conf %0.3f idx:%3d box[%0.4f,%0.4f,%0.4f,%0.4f] area %0.3f\n",i,
            classname,
            det->conf, det->index, det->x0, det->y0, det->x1, det->y1,
            (det->y1-det->y0)*(det->x1-det->x0));
            }
            log_trace("det %2d cls %8s conf %0.3f idx:%3d box[%0.4f,%0.4f,%0.4f,%0.4f] area %0.3f",i,
            classname,
            det->conf, det->index, det->x0, det->y0, det->x1, det->y1,
            (det->y1-det->y0)*(det->x1-det->x0));
    }
}
