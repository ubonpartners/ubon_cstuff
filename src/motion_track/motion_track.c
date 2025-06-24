#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cassert>
#include <algorithm>
#include <unistd.h>
#include <stdio.h>
#include "image.h"
#include "cuda_stuff.h"
#include "cuda_kernels.h"
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
    bool blur;
    float *noise_floor_device;
    uint8_t *row_masks_device, *row_masks_host;
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
    mt->max_width=yaml_get_int_value(yaml_base["motiontrack_max_width"], 320);    // must be <=64*8 = 512
    mt->max_height=yaml_get_int_value(yaml_base["motiontrack_max_height"], 320);  // must be <=512
    assert(mt->max_width>=64 && mt->max_width<=512 && ((mt->max_width&7)==0));
    assert(mt->max_height>=64 && mt->max_height<=512 && ((mt->max_height&7)==0));
    mt->blur=yaml_get_bool_value(yaml_base["motiontrack_blur"], true);
    mt->noise_floor_device=(float *)cuda_malloc(64*64*4);
    mt->row_masks_device=(uint8_t *)cuda_malloc(8*64);
    mt->row_masks_host=(uint8_t *)cuda_malloc_host(8*64);
    CHECK_CUDART_CALL(cudaMemset(mt->noise_floor_device, 0, 64*64*4));
    CHECK_CUDART_CALL(cudaMemset(mt->row_masks_device, 0, 64*8));
    return mt;
}

void motion_track_destroy(motion_track_t *mt)
{
    if (!mt) return;
    cuda_free(mt->noise_floor_device);
    cuda_free(mt->row_masks_device);
    cuda_free_host(mt->row_masks_host);
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
    if (mt->mad_delta==0)
    {
        mt->roi=ROI_ONE;
        return;
    }

    int scale_w, scale_h;
    determine_scale_size(img->width, img->height,
                         mt->max_width, mt->max_height, &scale_w, &scale_h,
                         10, 8, 8, false);

    image_t *image_scaled=image_scale_convert(img, IMAGE_FORMAT_YUV420_DEVICE, scale_w, scale_h);
    image_t *ref=mt->ref;

    assert((scale_w&7)==0);
    assert((scale_h&7)==0);

    if (mt->blur)
    {
        image_t *image_blurred=image_blur(image_scaled);
        if (image_blurred)
        {
            destroy_image(image_scaled);
            image_scaled=image_blurred;
        }
    }

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

        CHECK_CUDART_CALL(cudaMemset(mt->noise_floor_device, 0, 64*64*4));
        CHECK_CUDART_CALL(cudaMemset(mt->row_masks_device, 0, 64*8));

        mt->in_img=image_scaled;
        mt->block_w=image_scaled->width/8;
        mt->block_h=image_scaled->height/8;
        if (mt->noise_floor) free(mt->noise_floor);
        mt->noise_floor=(float *)malloc(sizeof(float)*mt->block_w*mt->block_h);
        memset(mt->noise_floor, 0, sizeof(float)*mt->block_w*mt->block_h);
        return;
    }

    image_t *mad_img=image_mad_4x4(mt->ref, image_scaled);
    assert(mad_img!=0);
    assert(mad_img->format==IMAGE_FORMAT_YUV420_DEVICE);

    //display_image("mad", mad_img);
    //usleep(250000);

    int block_w=mt->block_w;
    int block_h=mt->block_h;
    assert(block_w<=64);
    assert(mad_img->width==2*block_w && mad_img->height==2*block_h);
    float mad_delta=mt->mad_delta;
    float alpha=0.8f;
    float beta=0.95;

    cuda_generate_motion_mask(
        mad_img->y, mad_img->u, mad_img->v, mad_img->stride_y, mad_img->stride_uv,
        mt->noise_floor_device,
        block_w,block_h,
        mad_delta,alpha, beta,
        mt->row_masks_device,
        mad_img->stream);

    CHECK_CUDART_CALL(cudaMemcpyAsync(mt->row_masks_host, mt->row_masks_device, 64*8, cudaMemcpyDeviceToHost, mad_img->stream));
    CHECK_CUDART_CALL(cudaStreamSynchronize(mad_img->stream));
    destroy_image(mad_img);

    uint64_t *mp=(uint64_t *)mt->row_masks_host;
    uint64_t v_mask=0;
    uint64_t h_mask=0;
    for(int y=0;y<block_h;y++)
    {
        uint64_t mask=__builtin_bswap64(mp[y])>>(64-block_w);
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
        debugf("ROI %.3f %.3f %.3f %.3f A %0.4f",roi.box[0],roi.box[1],roi.box[2],roi.box[3], roi_area(&roi));
    }

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
