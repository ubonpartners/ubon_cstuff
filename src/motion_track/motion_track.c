#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cassert>
#include <algorithm>
#include <stdio.h>
#include "image.h"
#include "motion_track.h"
#include "display.h"
#include "roi.h"
#include "log.h"
#include "yaml_stuff.h"

#define debugf if (0) log_debug

struct motion_track
{
    int max_width, max_height;
    int block_w, block_h;
    float *noise_floor;
    float mad_delta;
    image_t *ref;
    image_t *in_img;
    roi_t roi;
};

motion_track_t *motion_track_create(const char *yaml_config)
{
    motion_track_t *mt=(motion_track_t *)malloc(sizeof(motion_track_t));
    memset(mt, 0, sizeof(motion_track_t));

    YAML::Node yaml_base=yaml_load(yaml_config);

    mt->mad_delta=yaml_get_float_value(yaml_base["motiontrack_mad_delta"], 24.0);
    mt->max_width=yaml_get_int_value(yaml_base["motiontrack_max_width"], 320);
    mt->max_height=yaml_get_int_value(yaml_base["motiontrack_max_height"], 320);
    return mt;
}

void motion_track_destroy(motion_track_t *mt)
{
    if (!mt) return;
    if (mt->ref) destroy_image(mt->ref);
    if (mt->in_img) destroy_image(mt->in_img);
    if (mt->noise_floor) free(mt->noise_floor);
    free(mt);
}

static int count_leading_zeros(uint64_t x) {
    return x ? __builtin_clzll(x) : 64;
}

static int count_trailing_zeros(uint64_t x) {
    return x ? __builtin_ctzll(x) : 64;
}

void motion_track_add_frame(motion_track_t *mt, image_t *img)
{
    int scale_w, scale_h;
    determine_scale_size(img->width, img->height,
                         mt->max_width, mt->max_height, &scale_w, &scale_h,
                         10, 8, 8, false);

    image_t *image_scaled=image_scale_convert(img, IMAGE_FORMAT_YUV420_DEVICE, scale_w, scale_h);
    image_t *ref=mt->ref;

    if (ref==0 || image_scaled->width!=ref->width || image_scaled->height!=ref->height)
    {
        if (ref)
        {
            destroy_image(mt->ref);
            mt->ref=ref=0;
        }
        if (mt->in_img)
        {
            destroy_image(mt->in_img);
            mt->in_img=0;
        }
        mt->roi.box[0]=0;
        mt->roi.box[1]=0;
        mt->roi.box[2]=1.0;
        mt->roi.box[3]=1.0;

        mt->in_img=image_scaled;
        mt->block_w=image_scaled->width/8;
        mt->block_h=image_scaled->height/8;
        if (mt->noise_floor) free(mt->noise_floor);
        mt->noise_floor=(float *)malloc(sizeof(float)*mt->block_w*mt->block_h);
        memset(mt->noise_floor, 0, sizeof(float)*mt->block_w*mt->block_h);
        return;
    }


    image_t *mad_img=image_mad_4x4(mt->ref, image_scaled);
    image_t *mad_img_host=image_convert(mad_img, IMAGE_FORMAT_YUV420_HOST);

    //display_image("MAD", mad_img);
    //display_image("REF", mt->ref);
    destroy_image(mad_img);
    image_sync(mad_img_host);

    int block_w=mt->block_w;
    int block_h=mt->block_h;
    assert(block_w<=64);
    float mad_delta=mt->mad_delta;
    uint64_t v_mask=0;
    uint64_t h_mask=0;
    for(int y=0;y<block_h;y++)
    {
        uint64_t mask=0;
        for(int x=0;x<block_w;x++)
        {
            uint8_t v0=mad_img_host->y[(x*2+0)+(y*2+0)*mad_img_host->stride_y];
            uint8_t v1=mad_img_host->y[(x*2+1)+(y*2+0)*mad_img_host->stride_y];
            uint8_t v2=mad_img_host->y[(x*2+0)+(y*2+1)*mad_img_host->stride_y];
            uint8_t v3=mad_img_host->y[(x*2+1)+(y*2+1)*mad_img_host->stride_y];
            uint8_t v4=mad_img_host->u[x+y*mad_img_host->stride_uv];
            uint8_t v5=mad_img_host->v[x+y*mad_img_host->stride_uv];
            uint8_t v=std::max(std::max(std::max(v0,v1), std::max(v2,v3)), std::max(v4,v5));
            float fv=(float)v;

            float nf=mt->noise_floor[x+y*block_w];
            float f=(fv<nf) ? 0.8f : 0.995f;
            nf=nf*f+v*(1.0-f);
            mt->noise_floor[x+y*block_w]=nf;

            bool motion=fv>nf+mad_delta;
            mask=(mask<<1)+(motion ? 1 : 0);
        }
        h_mask|=mask;
        v_mask=(v_mask<<1)+((mask==0) ? 0 : 1);
    }
    roi_t roi={0};
    if (h_mask!=0 && v_mask!=0)
    {
        float roi_l=8*count_leading_zeros(h_mask<<(64-block_w));
        float roi_r=image_scaled->width-8*count_trailing_zeros(h_mask);
        float roi_t=8*count_leading_zeros(v_mask<<(64-block_h));
        float roi_b=image_scaled->height-8*count_trailing_zeros(v_mask);
        roi.box[0]=roi_l/image_scaled->width;
        roi.box[1]=roi_t/image_scaled->height;
        roi.box[2]=roi_r/image_scaled->width;
        roi.box[3]=roi_b/image_scaled->height;
    }

    destroy_image(mad_img_host);
    mt->in_img=image_scaled;
    mt->roi=roi;
}

roi_t motion_track_get_roi(motion_track_t *mt)
{
    return mt->roi;
}

void motion_track_set_roi(motion_track_t *mt, roi_t roi)
{
    float a=roi_area(&roi);
    debugf("Mt set roi area %f\n",a);
    if (a>0.99)
    {
        if (mt->ref) destroy_image(mt->ref);
        mt->ref=mt->in_img;
        mt->in_img=0;
        return;
    }
    assert(mt->in_img!=0);
    if (a==0)
    {
        destroy_image(mt->in_img);
        mt->in_img=0;
        return;
    }
    int x0=int(roundf(roi.box[0]*mt->ref->width));
    int y0=int(roundf(roi.box[1]*mt->ref->height));
    int x1=int(roundf(roi.box[2]*mt->ref->width));
    int y1=int(roundf(roi.box[3]*mt->ref->height));
    x0=x0&(~1);
    x1=(x1+1)&(~1);
    y0=y0&(~1);
    y1=(y1+1)&(~1);
    //log_debug("Blend %d,%d->%d,%d\n",x0,y0,x1,y1);
    image_t *ref_new=image_blend(mt->ref, mt->in_img, x0, y0, x1-x0, y1-y0, x0, y0);
    destroy_image(mt->ref);
    destroy_image(mt->in_img);
    mt->ref=0;
    mt->in_img=0;
    mt->ref=ref_new;
}
