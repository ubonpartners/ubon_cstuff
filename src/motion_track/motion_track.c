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
#include "nvof.h"
#include "misc.h"

#define NVOF_SUPPORTED 1
#if (UBONCSTUFF_PLATFORM == 1) // Orin Nano
#undef NVOF_SUPPORTED
#endif
#define debugf if (0) log_debug

struct motion_track
{
    int max_width, max_height;
    int block_w, block_h;
    int frames_since_reset;
    float *noise_floor;
    float mad_delta;
    float alpha, beta; // alpha and beta for noise floor update
    float scene_change_sensitivity;
    bool blur;
    float *noise_floor_device;
    uint8_t *row_masks_device, *row_masks_host;
    image_t *ref;
    image_t *in_img;
    roi_t roi;
    bool scene_change;
    bool generate_of_results;
    of_results_t of_results;
    #ifdef NVOF_SUPPORTED
    nvof_t *nvof;
    #endif
};

motion_track_t *motion_track_create(const char *yaml_config)
{
    motion_track_t *mt=(motion_track_t *)malloc(sizeof(motion_track_t));
    memset(mt, 0, sizeof(motion_track_t));

    YAML::Node yaml_base=yaml_load(yaml_config);

    mt->mad_delta=yaml_get_float_value(yaml_base["motiontrack_mad_delta"], 24.0);
    mt->max_width=yaml_get_int_value(yaml_base["motiontrack_max_width"], 320);    // must be <=64*8 = 512
    mt->max_height=yaml_get_int_value(yaml_base["motiontrack_max_height"], 320);  // must be <=512
    mt->alpha=yaml_get_float_value(yaml_base["motiontrack_alpha"], 0.9);
    mt->beta=yaml_get_float_value(yaml_base["motiontrack_beta"], 0.995);
    mt->generate_of_results=yaml_get_bool_value(yaml_base["motiontrack_generate_of_results"], true);
    assert(mt->max_width>=64 && mt->max_width<=512 && ((mt->max_width&7)==0));
    assert(mt->max_height>=64 && mt->max_height<=512 && ((mt->max_height&7)==0));
    mt->blur=yaml_get_bool_value(yaml_base["motiontrack_blur"], true);
    mt->noise_floor_device=(float *)cuda_malloc(64*64*4);
    mt->row_masks_device=(uint8_t *)cuda_malloc(8*64);
    mt->row_masks_host=(uint8_t *)cuda_malloc_host(8*64);
    mt->scene_change_sensitivity=0.5f;
    #ifdef NVOF_SUPPORTED
    if (mt->generate_of_results) mt->nvof=nvof_create(mt, mt->max_width, mt->max_height);;
    #endif
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
    #ifdef NVOF_SUPPORTED
    if (mt->nvof) nvof_destroy(mt->nvof);
    #endif
    (mt);
}

void motion_track_reset(motion_track_t *mt)
{
    if (mt->ref)
    {
        destroy_image(mt->ref);
        mt->ref=0;
    }
    mt->roi=ROI_ONE;
    mt->scene_change=false;
    mt->frames_since_reset=0;
    #ifdef NVOF_SUPPORTED
    if (mt->nvof) nvof_reset(mt->nvof);
    #endif
}

static int count_leading_zeros(uint64_t x) {
    return x ? __builtin_clzll(x) : 64;
}

static int count_trailing_zeros(uint64_t x) {
    return x ? __builtin_ctzll(x) : 64;
}

bool motion_track_scene_change(motion_track_t *mt)
{
    return mt->scene_change;
}

static void motion_track_run_optical_flow(motion_track_t *mt, image_t *image_scaled)
{
    #ifdef NVOF_SUPPORTED
    if (mt->nvof)
    {
        nvof_results_t *r=nvof_execute(mt->nvof, image_scaled);
        if (r)
        {
            mt->of_results.grid_w=r->grid_w;
            mt->of_results.grid_h=r->grid_h;
            mt->of_results.flow=(oflow_vector_t*)r->flow;
            int total_cost=0;
            for(int y=0;y<r->grid_w*r->grid_h;y++)
            {
                total_cost+=std::max(0, r->costs[y]-5);
            }
            float avg_cost=(total_cost*1.0f)/(r->grid_w*r->grid_h);
            if (mt->scene_change_sensitivity!=0)
            {
                mt->scene_change=(avg_cost>16.0*(1.0-mt->scene_change_sensitivity))&&(mt->frames_since_reset>2);
                if (mt->scene_change) log_info("Scene change (t=%f %f %d %d)",image_scaled->time, avg_cost, total_cost, mt->frames_since_reset);
            }
        }
        else
        {
            mt->of_results.grid_w=0;
            mt->of_results.grid_h=0;
            mt->of_results.flow=0;
        }
    }
    #endif
}

void motion_track_add_frame(motion_track_t *mt, image_t *img)
{
    int scale_w, scale_h;

    if (file_trace_enabled)
    {
        FILE_TRACE("motiontracker add frame %dx%d fmt %d TS %f hash %lx",img->width,img->height,img->format,img->time, image_hash(img));
    }

    determine_scale_size(img->width, img->height,
                         mt->max_width, mt->max_height, &scale_w, &scale_h,
                         10, 8, 8, false);

    image_t *image_scaled=image_scale_convert(img, IMAGE_FORMAT_YUV420_DEVICE, scale_w, scale_h);
    image_t *ref=mt->ref;

    FILE_TRACE("MT scaled size %dx%d",scale_w,scale_h);

    assert((scale_w&7)==0);
    assert((scale_h&7)==0);
    mt->scene_change=false;

    if (mt->mad_delta==0)
    {
        motion_track_run_optical_flow(mt, image_scaled);
        mt->roi=ROI_ONE;
        if (mt->in_img)
        {
            destroy_image(mt->in_img);
            mt->in_img=0;
        }
        mt->in_img=image_scaled;
        return;
    }

    image_t *nvof_image=image_reference(image_scaled);
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
        mt->frames_since_reset=0;
        motion_track_run_optical_flow(mt, nvof_image);
        destroy_image(nvof_image);
        return;
    }

    image_t *mad_img=image_mad_4x4(mt->ref, image_scaled);
    assert(mad_img!=0);
    assert(mad_img->format==IMAGE_FORMAT_YUV420_DEVICE);

    if (file_trace_enabled)
    {
        FILE_TRACE("motiontracker REF %lx SCAL %lx MAD %lx",image_hash(mt->ref), image_hash(image_scaled), image_hash(mad_img));
    }

    //display_image("mad", mad_img);
    //usleep(250000);

    int block_w=mt->block_w;
    int block_h=mt->block_h;
    assert(block_w<=64);
    assert(mad_img->width==2*block_w && mad_img->height==2*block_h);
    float mad_delta=mt->mad_delta;

    // noise floor is tracked by updating from new MAD for each block
    // if new noise<noise measure : noise_measure=alpha*noise_measure+(1-alpha)*new_noise - fast fall
    // if new noise>noise measure : noise_measure=beta*noise_measure+(1-beta)*new_noise - slow rise

    float alpha=mt->alpha;
    float beta=mt->beta;
    int alpha_ramp_frames=16;
    if (mt->frames_since_reset<alpha_ramp_frames)
    {
        alpha=(alpha*mt->frames_since_reset)/alpha_ramp_frames;
    }

    cuda_generate_motion_mask(
        mad_img->y, mad_img->u, mad_img->v, mad_img->stride_y, mad_img->stride_uv,
        mt->noise_floor_device,
        block_w,block_h,
        mad_delta,alpha, beta,
        mt->row_masks_device,
        mad_img->stream);

    CHECK_CUDART_CALL(cudaStreamSynchronize(mad_img->stream));
    CHECK_CUDART_CALL(cudaMemcpy(mt->row_masks_host, mt->row_masks_device, 64*8, cudaMemcpyDeviceToHost));

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
    assert(roi.box[0]>=0.0f && roi.box[2]<=1.0f && roi.box[1]>=0.0f && roi.box[3]<=1.0f);

    // run optical flow

    if (roi_area(&roi)>0.01)
        motion_track_run_optical_flow(mt, nvof_image);
    #ifdef NVOF_SUPPORTED
    else if (mt->nvof)
        nvof_set_no_motion(mt->nvof); // skip running NVOF, frames are too similar
    #endif

    destroy_image(nvof_image);
    mt->in_img=image_scaled;
    mt->roi=roi;
    mt->frames_since_reset++;
}

roi_t motion_track_get_roi(motion_track_t *mt)
{
    return mt->roi;
}

void motion_track_set_roi(motion_track_t *mt, roi_t roi)
{
    float a=roi_area(&roi);
    debugf("Mt set roi area %f\n",a);

    if (file_trace_enabled)
    {
        FILE_TRACE("motion_track_set_roi %0.4f,%0.4f,%0.4f,%0.4f", roi.box[0], roi.box[1],roi.box[2],roi.box[3]);
    }

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

of_results_t *motion_track_get_of_results(motion_track_t *mt)
{
    return &mt->of_results;
}

static void motion_track_predict_delta(motion_track_t *mt, float *pt, float *d)
{
    if ((mt->ref==0)||(mt->of_results.grid_w==0))
    {
        d[0]=0;
        d[1]=0;
        return;
    }
    int grid_w=mt->of_results.grid_w;
    int grid_h=mt->of_results.grid_h;
    float x=pt[0];
    float y=pt[1];
    int ix=std::min(grid_w-1, std::max(0, (int)(x*grid_w+0.5f)));
    int iy=std::min(grid_h-1, std::max(0, (int)(y*grid_h+0.5f)));
    int offs=ix+iy*grid_w;
    oflow_vector_t *fv=mt->of_results.flow+offs;
    float dx=fv->flowx/(4.0f*32.0f*grid_w);
    float dy=fv->flowy/(4.0f*32.0f*grid_h);
    d[0]=dx;
    d[1]=dy;
}

void motion_track_predict_point_inplace(motion_track_t *mt, float *pt)
{
    float d[2];
    motion_track_predict_delta(mt, pt, d);
    pt[0]=std::max(0.0f, std::min(1.0f, pt[0]-d[0]));
    pt[1]=std::max(0.0f, std::min(1.0f, pt[1]-d[1]));
}

typedef struct sample
{
    float xf, yf, w;
} sample_t;

void motion_track_predict_box_inplace(motion_track_t *mt, float *box)
{
    if (!mt) return;

    static const sample_t samples[] = {
        {0.5f, 0.5f, 0.5f},
        {0.35f, 0.5f, 0.125f},
        {0.65f, 0.5f, 0.125f},
        {0.5f, 0.35f, 0.125f},
        {0.5f, 0.65f, 0.125f}
    };
    float dx=0;
    float dy=0;
    float x0=box[0];
    float y0=box[1];
    float x1=box[2];
    float y1=box[3];
    for(int i=0;i<5;i++)
    {
        float xf=samples[i].xf;
        float yf=samples[i].yf;
        float p[2]={x0*xf+x1*(1.0f-xf), y0*yf+y1*(1.0f-yf)};
        float d[2];
        motion_track_predict_delta(mt, p, d);
        dx+=samples[i].w*d[0];
        dy+=samples[i].w*d[1];
    }
    box[0]=std::max(0.0f, std::min(1.0f, x0-dx));
    box[1]=std::max(0.0f, std::min(1.0f, y0-dy));
    box[2]=std::max(0.0f, std::min(1.0f, x1-dx));
    box[3]=std::max(0.0f, std::min(1.0f, y1-dy));
}
