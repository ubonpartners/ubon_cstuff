#include <stdint.h>
#include <string.h>
#include <cassert>
#include <mutex>
#include <functional>
#include "utrack.h"
#include "log.h"
#include "yaml_stuff.h"
#include "memory_stuff.h"
#include "maths_stuff.h"
#include "match.h"
#include "misc.h"
#include "kalman_tracker.h"
#include "detections.h"


#define REID_VECTOR_LEN 80
#define MAX_TRACKED     350
#define debugf if (0) log_warn

typedef enum trackstate
{
    TRACKSTATE_NEW = 0,
    TRACKSTATE_TRACKED = 1,
    TRACKSTATE_LOST = 2,
    TRACKSTATE_REMOVED = 3
} trackstate_t;

typedef struct utdet
{
    detection_t det;
    bool matched;
    uint32_t observations;
    uint32_t num_missed;
    float adjusted_confidence;
    trackstate_t track_state;
    double time;
    double last_detect_time;
    KalmanBoxTracker *kf;
    float vbox[4]; // vicinity box, for tracked objects it's an elarged box encompassing the general area the object is
    float kf_predicted_box[4];
    float of_predicted_box[4];
    float reid_norm[REID_VECTOR_LEN];
} utdet_t;

struct utrack
{
    float param_immediate_confirm_thr;
    float param_track_initial_thr;
    float param_track_high_thr;
    float param_track_low_thr;
    float param_new_track_thr;
    float param_person_det_thr_single_frame;
    float param_vbox_expand;
    float param_match_thr_initial;
    float param_match_thr_high;
    float param_match_thr_low;
    float param_face_weight;
    float param_kf_weight;
    float param_kf_warmup;
    float param_sim_weight;
    float param_delete_dup_iou;
    float param_track_buffer_seconds;
    float param_fuse_scores;
    float param_pose_conf;
    float param_roi_expand_ratio;
    bool  param_simple;

    uint64_t next_track_id;
    int num_tracked;
    utdet_t *det[MAX_TRACKED];
    utdet_t *tracked[MAX_TRACKED];
};

static std::once_flag initFlag;
static block_allocator_t *utdet_allocator=0;


static void do_utrack_init()
{
    log_debug("Utrack init");
    utdet_allocator=block_allocator_create("utdet_allocator allocator", sizeof(utdet_t));
}

utrack_t *utrack_create(const char *yaml_config)
{
    std::call_once(initFlag, do_utrack_init);
    utrack_t *ut=(utrack_t *)malloc(sizeof(utrack_t));
    memset(ut, 0, sizeof(utrack_t));
    YAML::Node yaml_base=yaml_load(yaml_config)["utrack"];

    ut->next_track_id=0xbeef0000;

    ut->param_immediate_confirm_thr=yaml_get_float_value(yaml_base["immediate_confirm_thr"], 1.18f);
    ut->param_track_initial_thr=yaml_get_float_value(yaml_base["track_initial_thr"], 0.495f);
    ut->param_track_high_thr=yaml_get_float_value(yaml_base["track_high_thr"], 0.9f);
    ut->param_track_low_thr=yaml_get_float_value(yaml_base["track_low_thr"], 0.115f);
    ut->param_new_track_thr=yaml_get_float_value(yaml_base["new_track_thr"], 0.57f);
    ut->param_person_det_thr_single_frame=yaml_get_float_value(yaml_base["person_det_thr_single_frame"], 0.05f);
    ut->param_vbox_expand=yaml_get_float_value(yaml_base["vbox_expand"], 0.1f);
    ut->param_match_thr_initial=yaml_get_float_value(yaml_base["match_thr_initial"], 0.66f);
    ut->param_match_thr_high=yaml_get_float_value(yaml_base["match_thr_high"], 0.225f);
    ut->param_match_thr_low=yaml_get_float_value(yaml_base["match_thr_low"], 0.022f);
    ut->param_face_weight=yaml_get_float_value(yaml_base["face_weight"], 0.020f);
    ut->param_kf_weight=yaml_get_float_value(yaml_base["kf_weight"], 1.0f);
    ut->param_kf_warmup=yaml_get_float_value(yaml_base["kf_warmup"], 0.7f);
    ut->param_sim_weight=yaml_get_float_value(yaml_base["sim_weight"], 0.2f);
    ut->param_delete_dup_iou=yaml_get_float_value(yaml_base["delete_dup_iou"], 0.82f);
    ut->param_track_buffer_seconds=yaml_get_float_value(yaml_base["track_buffer_seconds"], 2.0f);
    ut->param_fuse_scores=yaml_get_float_value(yaml_base["fuse_scores"], 0.94f);
    ut->param_pose_conf=yaml_get_float_value(yaml_base["pose_conf"], 0.004f);
    ut->param_roi_expand_ratio=yaml_get_float_value(yaml_base["roi_expand_ratio"], 0.05f);
    ut->param_simple=yaml_get_bool_value(yaml_base["simple"], false);
    if (ut->param_simple) printf("SIMPLE %d\n",ut->param_simple);

    return ut;
}

void utdet_destroy(utdet_t *utdet)
{
    detection_destroy((detection_t *)utdet);
}

void utrack_destroy(utrack_t *ut)
{
    if (!ut) return;
    for(int i=0;i<ut->num_tracked;i++) utdet_destroy(ut->tracked[i]);
    free(ut);
}

static void utrack_normalize_reid_vectors(utdet_t **dets, int num_det, utdet_t **tracked, int num_tracked)
{
    // compute the mean-normalized REID embeddings
    // we subtract the mean of all the detections+tracked dets, then L2-norm the
    // resulting vector. This gives much better results vs raw embeddings
    float reid_mean[REID_VECTOR_LEN];
    memset(reid_mean, 0, 64*sizeof(float));
    for(int i=0;i<num_det;i++) vec_accum(dets[i]->det.reid, reid_mean, REID_VECTOR_LEN);
    for(int i=0;i<num_tracked;i++) vec_accum(tracked[i]->det.reid, reid_mean, REID_VECTOR_LEN);
    vec_scale(reid_mean, 1.0/(num_det+num_tracked+1e-7), REID_VECTOR_LEN);
    for(int i=0;i<num_det;i++) vec_mean_normalize(dets[i]->det.reid, reid_mean, dets[i]->reid_norm, REID_VECTOR_LEN);
    for(int i=0;i<num_tracked;i++) vec_mean_normalize(tracked[i]->det.reid, reid_mean, tracked[i]->reid_norm, REID_VECTOR_LEN);
}

typedef struct match_context
{
    float match_thr;
    float param_kf_weight;
    float param_kf_warmup;
    float param_fuse_scores;
    float param_sim_weight;
    bool param_simple;
} match_context_t;


static bool intersect(const detection_t *da, float *db)
{
    float x_left = fmaxf(da->x0, db[0]);
    float y_top = fmaxf(da->y0, db[1]);
    float x_right = fminf(da->x1, db[2]);
    float y_bottom = fminf(da->y1, db[3]);

    float inter=(x_right - x_left) * (y_bottom - y_top);
    return inter>0;
}

static float iou(const detection_t *da, float *db)
{
    float x_left = fmaxf(da->x0, db[0]);
    float y_top = fmaxf(da->y0, db[1]);
    float x_right = fminf(da->x1, db[2]);
    float y_bottom = fminf(da->y1, db[3]);

    if (x_right < x_left || y_bottom < y_top)
        return 0.0f;

    float inter=(x_right - x_left) * (y_bottom - y_top);

    float area_a = (da->x1 - da->x0) * (da->y1 - da->y0);
    float area_b = (db[2] - db[0]) * (db[3] - db[1]);

    float union_area = area_a + area_b - inter;

    if (union_area <= 0.0f)
        return 0.0f;

    float ret=inter / union_area;
    return ret;
}

static float match_cost(const detection_t *det_new, const detection_t *det_existing, void *ctx)
{
    match_context_t *mc=(match_context_t *)ctx;
    utdet_t *tdet_existing=(utdet_t *)det_existing;
    utdet_t *tdet_new=(utdet_t *)det_new;

    if (false==intersect(det_new, tdet_existing->vbox)) return 0.0f;

    float kf_weight=mc->param_kf_weight;
    float of_weight=1.0;

    float kf_score=iou(det_new, tdet_existing->kf_predicted_box);
    float of_score=iou(det_new, tdet_existing->of_predicted_box);
    //if (mc->param_simple) return (of_score+kf_score)/2;

    //if ((of_score+kf_score)==0) return 0.0f;

    float sim=vec_dot(tdet_existing->reid_norm, tdet_new->reid_norm, REID_VECTOR_LEN);

    sim*=mc->param_sim_weight;

    if (tdet_existing->observations<2)
        kf_weight=0;
    else
    {
        float f=powf(std::max(0.1f, 1.0f/tdet_existing->observations ), mc->param_kf_warmup);
        kf_weight*=(1-f);
    }
    float score=(of_score*of_weight+kf_score*kf_weight)/(of_weight+kf_weight);
    score+=sim;
    if (score<mc->match_thr) return 0;

    score=score*powf(det_new->conf, mc->param_fuse_scores);

    return score;
}

static float lost_object_match_score(const detection_t *self, const detection_t *other, void *ctx)
{
    utdet_t *tdet_self=(utdet_t *)self;
    utdet_t *tdet_other=(utdet_t *)other;
    if (tdet_self->track_state!=TRACKSTATE_TRACKED) return 0;
    if (tdet_other->track_state!=TRACKSTATE_LOST) return 0;
    float box_other[4];
    box_other[0]=tdet_other->det.x0;
    box_other[1]=tdet_other->det.y0;
    box_other[2]=tdet_other->det.x1;
    box_other[3]=tdet_other->det.y1;
    float box_score=iou(self, box_other);
    float thr=*((float *)ctx);
    if (box_score<thr) return 0;
    return box_score;
}

roi_t utrack_predict_positions(utrack_t *ut, double rtp_time, motion_track_t *mt, roi_t motion_roi)
{
    float min_x=motion_roi.box[0];
    float min_y=motion_roi.box[1];
    float max_x=motion_roi.box[2];
    float max_y=motion_roi.box[3];

    FILE_TRACE("predict positions time %f", rtp_time);

    debugf("utrack_predict_positions");

    FILE_TRACE("Predicted (Motion ROI) [%0.4f,%0.4f,%0.4f,%0.4f]",min_x,min_y,max_x,max_y);

    utdet_t **tracked=ut->tracked;
    int num_tracked=ut->num_tracked;
    for(int i=0;i<num_tracked;i++)
    {
        utdet_t *tdet=tracked[i];
        assert(tdet->kf!=0);
        Vector4f v=tdet->kf->predict(rtp_time);
        tdet->kf_predicted_box[0]=v[0];
        tdet->kf_predicted_box[1]=v[1];
        tdet->kf_predicted_box[2]=v[2];
        tdet->kf_predicted_box[3]=v[3];

        min_x=std::min(min_x, v[0]);
        min_y=std::min(min_y, v[1]);
        max_x=std::max(max_x, v[2]);
        max_y=std::max(max_y, v[3]);

        tdet->of_predicted_box[0]=tdet->det.x0;
        tdet->of_predicted_box[1]=tdet->det.y0;
        tdet->of_predicted_box[2]=tdet->det.x1;
        tdet->of_predicted_box[3]=tdet->det.y1;

        min_x=std::min(min_x, tdet->of_predicted_box[0]);
        min_y=std::min(min_y, tdet->of_predicted_box[1]);
        max_x=std::max(max_x, tdet->of_predicted_box[2]);
        max_y=std::max(max_y, tdet->of_predicted_box[3]);

        // predict box motion by some random points
        motion_track_predict_box_inplace(mt, tdet->of_predicted_box);
        // enlarge predicted box by predicted motion of pose points, if any
        for(int j=0;j<tdet->det.num_pose_points;j++)
        {
            if (tdet->det.pose_points[j].conf>0.05f)
            {
                float pt[2];
                pt[0]=tdet->det.pose_points[j].x;
                pt[1]=tdet->det.pose_points[j].y;
                motion_track_predict_point_inplace(mt, pt);
                tdet->of_predicted_box[0]=std::min(tdet->of_predicted_box[0], pt[0]);
                tdet->of_predicted_box[1]=std::min(tdet->of_predicted_box[1], pt[1]);
                tdet->of_predicted_box[2]=std::max(tdet->of_predicted_box[2], pt[0]);
                tdet->of_predicted_box[3]=std::max(tdet->of_predicted_box[3], pt[1]);
            }
        }
        min_x=std::min(min_x, tdet->of_predicted_box[0]);
        min_y=std::min(min_y, tdet->of_predicted_box[1]);
        max_x=std::max(max_x, tdet->of_predicted_box[2]);
        max_y=std::max(max_y, tdet->of_predicted_box[3]);

        FILE_TRACE("det bx [%.4f %.4f %.4f %.4f] OF [%.4f %.4f %.4f %.4f] KF [%.4f %.4f %.4f %.4f]",
            tdet->det.x0,tdet->det.y0,tdet->det.x1,tdet->det.y1,
            tdet->of_predicted_box[0], tdet->of_predicted_box[1], tdet->of_predicted_box[2], tdet->of_predicted_box[3],
            tdet->kf_predicted_box[0], tdet->kf_predicted_box[1], tdet->kf_predicted_box[2], tdet->kf_predicted_box[3]);

    }

    FILE_TRACE("Predicted position ROI (pre expand) [%0.4f,%0.4f,%0.4f,%0.4f]",min_x,min_y,max_x,max_y);

    float e_w=std::max(0.05f, max_x-min_x);
    float e_h=std::max(0.05f, max_y-min_y);
    float param_roi_expand_ratio=ut->param_roi_expand_ratio;
    min_x=std::max(0.0f, std::min(1.0f, min_x-param_roi_expand_ratio*0.5f*e_w));
    min_y=std::max(0.0f, std::min(1.0f, min_y-param_roi_expand_ratio*0.5f*e_h));
    max_x=std::max(0.0f, std::min(1.0f, max_x+param_roi_expand_ratio*0.5f*e_w));
    max_y=std::max(0.0f, std::min(1.0f, max_y+param_roi_expand_ratio*0.5f*e_h));

    roi_t roi;
    roi.box[0]=min_x;
    roi.box[1]=min_y;
    roi.box[2]=max_x;
    roi.box[3]=max_y;
    FILE_TRACE("Predicted (TS %f) position ROI (post expand %0.3f) [%0.4f,%0.4f,%0.4f,%0.4f] ",rtp_time,roi.box[0],roi.box[1],roi.box[2],roi.box[3]);

    if (ut->param_simple)
    {
        roi.box[0]=0;
        roi.box[1]=0;
        roi.box[2]=1;
        roi.box[3]=1;
    }
    return roi;
}

static int compare_utdet_desc(const void *a, const void *b) {
    const utdet_t *ua = *(const utdet_t **)a;
    const utdet_t *ub = *(const utdet_t **)b;
    if (ua->adjusted_confidence < ub->adjusted_confidence) return 1;
    if (ua->adjusted_confidence > ub->adjusted_confidence) return -1;
    return 0;
}

void utrack_reset(utrack_t *ut)
{
    for(int i=0;i<ut->num_tracked;i++)
    {
        utdet_t *tdet=ut->tracked[i];
        utdet_destroy(tdet);
        ut->tracked[i]=0;
    }
    ut->num_tracked=0;
}

detection_list_t *utrack_run(utrack_t *ut, detection_list_t *dets_in, double rtp_time, bool single_frame)
{
    FILE_TRACE("==================================");
    FILE_TRACE("utrack run: time %f; %d detections", rtp_time, dets_in->num_detections);

    utdet_t *output_objects[MAX_TRACKED];
    utdet_t **dets=ut->det;
    utdet_t **tracked=ut->tracked;
    int num_det=0;
    int num_tracked=ut->num_tracked;
    int num_output_objects=0;

    // extend dets_in to utdet
    assert(dets_in->num_detections<=MAX_TRACKED);
    float pose_conf=ut->param_pose_conf;
    for(int i=0;i<dets_in->num_detections;i++)
    {
        detection_t *det=dets_in->det[i];
        if (det->cl!=0) continue;
        utdet_t *utdet=(utdet_t *)block_alloc(utdet_allocator);
        memset(utdet, 0, sizeof(utdet_t));
        memcpy(&utdet->det, det, sizeof(detection_t));
        utdet->det.track_id=0xdeaddead;

        utdet->observations=1;
        utdet->kf=0;
        float pose=0;
        for(int i=0;i<det->num_pose_points;i++) pose+=det->pose_points[i].conf;
        utdet->adjusted_confidence=utdet->det.conf+pose*pose_conf;
        if (true) // expand by pose
        {
            for(int i=0;i<det->num_pose_points;i++)
            {
                if (det->pose_points[i].conf>0.05)
                {
                    utdet->det.x0=std::min(utdet->det.x0, det->pose_points[i].x);
                    utdet->det.y0=std::min(utdet->det.y0, det->pose_points[i].y);
                    utdet->det.x1=std::max(utdet->det.x1, det->pose_points[i].x);
                    utdet->det.y1=std::max(utdet->det.y1, det->pose_points[i].y);
                }
            }
        }
        utdet->det.overlap_mask=box_to_8x8_mask(utdet->det.x0, utdet->det.y0, utdet->det.x1, utdet->det.y1);
        //printf("adjusted %f (pose %f %f)\n",utdet->adjusted_confidence, pose, pose_conf);
        utdet->track_state=TRACKSTATE_NEW;
        utdet->matched=false;
        dets[num_det++]=utdet;
    }

    qsort(dets, num_det, sizeof(utdet_t *), compare_utdet_desc);

    for(int i=0;i<num_det;i++)
    {
        utdet_t *utdet=dets[i];
        FILE_TRACE("Initial det %2d msk %016lx conf %0.4f box [%0.4f,%0.4f,%0.4f,%0.4f]", i,
                    utdet->det.overlap_mask, utdet->adjusted_confidence,
                    utdet->det.x0, utdet->det.y0, utdet->det.x1, utdet->det.y1);
    }

    if (single_frame)
    {
        for(int i=0;i<num_tracked;i++)
        {
            utdet_t *tdet=tracked[i];
            utdet_destroy(tdet);
            tracked[i]=0;
        }
        num_tracked=0;
    }

    for(int i=0;i<num_tracked;i++)
    {
        utdet_t *tdet=tracked[i];
        tdet->matched=false;
        tdet->time=rtp_time;
        // regenerate overlap mask as det has probably moved (will be used a lot...)
        float vbox_x0=std::min(tdet->det.x0, std::min(tdet->of_predicted_box[0], tdet->kf_predicted_box[0]));
        float vbox_y0=std::min(tdet->det.y0, std::min(tdet->of_predicted_box[1], tdet->kf_predicted_box[1]));
        float vbox_x1=std::max(tdet->det.x1, std::max(tdet->of_predicted_box[2], tdet->kf_predicted_box[2]));
        float vbox_y1=std::max(tdet->det.y1, std::max(tdet->of_predicted_box[3], tdet->kf_predicted_box[3]));
        float w=vbox_x1-vbox_x0;
        float h=vbox_y1-vbox_y0;
        float e=ut->param_vbox_expand;
        vbox_x0-=(0.5*e*w);
        vbox_y0-=(0.5*e*h);
        vbox_x1+=(0.5*e*w);
        vbox_y1+=(0.5*e*h);
        tdet->vbox[0]=vbox_x0;
        tdet->vbox[1]=vbox_y0;
        tdet->vbox[2]=vbox_x1;
        tdet->vbox[3]=vbox_y1;
        tdet->det.overlap_mask=box_to_8x8_mask(vbox_x0, vbox_y0, vbox_x1, vbox_y1);
        FILE_TRACE(" MASKGEN %0.4f %0.4f %0.4f %0.4f %016lx", vbox_x0,vbox_y0,vbox_x1,vbox_y1,tdet->det.overlap_mask);
    }

    for(int i=0;i<num_tracked;i++)
    {
        utdet_t *tdet=tracked[i];
        FILE_TRACE("Initial tracked %2d %lx msk %016lx [%.4f %.4f %.4f %.4f] OF [%.4f %.4f %.4f %.4f] KF [%.4f %.4f %.4f %.4f]",
            i, tdet->det.track_id, tdet->det.overlap_mask, tdet->det.x0,tdet->det.y0,tdet->det.x1,tdet->det.y1,
            tdet->of_predicted_box[0], tdet->of_predicted_box[1], tdet->of_predicted_box[2], tdet->of_predicted_box[3],
            tdet->kf_predicted_box[0], tdet->kf_predicted_box[1], tdet->kf_predicted_box[2], tdet->kf_predicted_box[3]);
    }

    utrack_normalize_reid_vectors(dets, num_det, tracked, num_tracked);

    // ================================================================================================================
    // matching

    match_context_t mc;
    mc.param_kf_weight=ut->param_kf_weight;
    mc.param_kf_warmup=ut->param_kf_warmup;
    mc.param_fuse_scores=ut->param_fuse_scores;
    mc.param_sim_weight=ut->param_sim_weight;
    mc.param_simple=ut->param_simple;

    for(int pass=0;((pass<3) && (single_frame==false));pass++)
    {
        debugf("Start pass %d",pass);
        std::function<bool(utdet_t *)> det_filter_func, track_filter_func;

        if (pass==0)
        {
            det_filter_func=[ut](utdet_t* x){return x->matched==false && x->adjusted_confidence>ut->param_track_initial_thr; };
            track_filter_func=[ut](utdet_t* x){return x->matched==false && x->track_state!=TRACKSTATE_LOST; };
            mc.match_thr=ut->param_match_thr_initial;
        }
        else if (pass==1)
        {
            det_filter_func=[ut](utdet_t* x){return x->matched==false && x->adjusted_confidence>ut->param_track_high_thr; };
            track_filter_func=[ut](utdet_t* x){return x->matched==false;};
            mc.match_thr=ut->param_match_thr_high;
        }
        else // pass==2
        {
            det_filter_func=[ut](utdet_t* x){return x->matched==false && x->adjusted_confidence>ut->param_track_low_thr; };
            track_filter_func=[ut](utdet_t* x){return x->matched==false;};
            mc.match_thr=ut->param_match_thr_low;
        }

        utdet_t *det_filtered[num_det];
        utdet_t *tracked_filtered[num_tracked];
        int num_det_filtered=0;
        int num_tracked_filtered=0;

        for(int i=0;i<num_det;i++)
        {
            utdet_t *det=dets[i];
            if (det_filter_func(det)) det_filtered[num_det_filtered++]=det;
        }
        for(int i=0;i<num_tracked;i++)
        {
            utdet_t *det=tracked[i];
            if (track_filter_func(det)) tracked_filtered[num_tracked_filtered++]=det;
        }

        if ((num_det_filtered==0)||(num_tracked_filtered==0)) continue;

        debugf("Running match pass %d; %d dets, %d tracked",pass, num_det_filtered, num_tracked_filtered);

        int maxn=1+std::min(num_det_filtered, num_tracked_filtered);
        uint16_t match_ind_det[maxn];
        uint16_t match_ind_tracked[maxn];
        float match_score[maxn];

        int n=match_detections_greedy((detection_t **)det_filtered, num_det_filtered,
                                    (detection_t **)tracked_filtered, num_tracked_filtered,
                                    match_cost, &mc,
                                    match_ind_det, match_ind_tracked, match_score, false);

        for(int i=0;i<n;i++)
        {
            utdet_t *new_obj=det_filtered[match_ind_det[i]];
            utdet_t *old_obj=tracked_filtered[match_ind_tracked[i]];
            float score=match_score[i];
            FILE_TRACE("match pass %d:%d/%d Match %2d<->%2d score %1.4f",pass,i,n,
                        match_ind_det[i], match_ind_tracked[i], score);
            debugf("Match pass %d : Match to old obj %lx conf %0.3f",pass,old_obj->det.track_id, new_obj->adjusted_confidence);
            new_obj->det.track_id=old_obj->det.track_id;
            new_obj->observations=old_obj->observations+1;
            old_obj->matched=true;
            new_obj->matched=true;
            assert(new_obj->kf==0);
            new_obj->kf=old_obj->kf;
            new_obj->last_detect_time=rtp_time;
            new_obj->num_missed=0;
            old_obj->kf=0;
            Vector4f v(new_obj->det.x0, new_obj->det.y0, new_obj->det.x1, new_obj->det.y1);
            new_obj->kf->update(v, rtp_time);

            float ema=0.5;
            for(int i=0;i<REID_VECTOR_LEN;i++) new_obj->det.reid[i]=ema*new_obj->det.reid[i]+(1.0-ema)*old_obj->det.reid[i];
            new_obj->det.fiqa_score=0.5*new_obj->det.fiqa_score+0.5*old_obj->det.fiqa_score;
            if ((pass==0)||(pass==1))
                new_obj->track_state=TRACKSTATE_TRACKED;
            else if (old_obj->track_state==TRACKSTATE_LOST)
                new_obj->track_state=TRACKSTATE_TRACKED;
            else
                new_obj->track_state=old_obj->track_state;

            assert(num_output_objects<MAX_TRACKED);
            output_objects[num_output_objects++]=(utdet_t *)block_reference(new_obj);
        }
    }
    debugf("All matching done");

    // ================================================================================================================
    // deal with new objects that don't match any existing objects

    for(int i=0;i<num_det;i++)
    {
        utdet_t *det=dets[i];
        if (det->matched) continue;
        debugf("Unmatched new obj %lx conf %0.3f", det->det.track_id, det->det.conf);
        float thr=(single_frame) ? ut->param_person_det_thr_single_frame : ut->param_new_track_thr;
        if (det->adjusted_confidence>thr)
        {
            assert(det->det.track_id==0xdeaddead);
            det->det.track_id=ut->next_track_id++;
            Vector4f v(det->det.x0, det->det.y0, det->det.x1, det->det.y1);
            det->kf=new KalmanBoxTracker(v, rtp_time);
            det->last_detect_time=rtp_time;
            det->num_missed=0;
            if ((det->adjusted_confidence>ut->param_immediate_confirm_thr)||single_frame)
                det->track_state=TRACKSTATE_TRACKED;
            assert(num_output_objects<MAX_TRACKED);
            output_objects[num_output_objects++]=(utdet_t *)block_reference(det);
        }
    }

    // done with detections now
    for(int i=0;i<num_det;i++)
    {
        utdet_destroy(dets[i]);
        dets[i]=0;
    }

    // ================================================================================================================
    // determine which objects to delete
    for(int i=0;i<num_tracked;i++)
    {
        utdet_t *tdet=tracked[i];
        if (tdet->matched)
        {
            // got replaced by a matched new object
            if (tdet->kf) delete tdet->kf;
            utdet_destroy(tdet);
            continue;
        }
        tdet->num_missed++;
        double time_since_detection=rtp_time-tdet->last_detect_time;
        bool keep=time_since_detection<ut->param_track_buffer_seconds && tdet->num_missed<10;
        keep=keep||(tdet->track_state==TRACKSTATE_TRACKED);
        keep&=(num_output_objects<MAX_TRACKED);
        if (keep)
        {
            assert(tdet!=0);
            assert(num_output_objects<MAX_TRACKED);
            output_objects[num_output_objects++]=tdet;
        }
        else
        {
            tdet->track_state=TRACKSTATE_REMOVED;
            debugf("Deleting object %p:%d:%lx",tdet,i,tdet->det.track_id);
            if (tdet->kf) delete tdet->kf;
            utdet_destroy(tdet);
            tracked[i]=0;
        }
    }

    // update tracked object list
    assert(ut->tracked==tracked);
    memcpy(ut->tracked, output_objects, num_output_objects*sizeof(utdet_t*));
    ut->num_tracked=num_tracked=num_output_objects;

    // ================================================================================================================
    // determine "lost" objects"
    for(int i=0;i<num_tracked;i++)
    {
        if (tracked[i]->track_state==TRACKSTATE_TRACKED)
        {
            if (tracked[i]->num_missed>=2) tracked[i]->track_state=TRACKSTATE_LOST;
        }
        debugf("final tracked list %d:%lx state %d",i,tracked[i]->det.track_id, tracked[i]->track_state);
    }

    // ================================================================================================================
    // remove duplicated objects

    uint16_t new_ind[num_tracked], old_ind[num_tracked];
    int n=match_detections_greedy((detection_t **)tracked, num_tracked,
                                 (detection_t **)tracked, num_tracked,
                                 lost_object_match_score, &ut->param_delete_dup_iou,
                                 new_ind, old_ind);
    for (int i=0;i<n;i++)
    {
        assert(tracked[old_ind[i]]->track_state==TRACKSTATE_LOST);
        tracked[old_ind[i]]->track_state=TRACKSTATE_REMOVED;
    }

    int num_not_removed=0;
    for(int i=0;i<num_tracked;i++)
    {
        utdet_t *tdet=tracked[i];
        if (tdet->track_state!=TRACKSTATE_REMOVED)
            tracked[num_not_removed++]=tdet;
        else
        {
            if (tdet->kf) delete tdet->kf;
            utdet_destroy(tdet);
        }
    }
    num_tracked=ut->num_tracked=num_not_removed;

    //================================================================================================================
    // output objects

    detection_list_t *out_list=detection_list_create(num_tracked);
    out_list->time=rtp_time;
    out_list->md=dets_in->md;
    out_list->num_detections=0;
    for(int i=0;i<num_tracked;i++)
    {
        utdet_t *tdet=tracked[i];
        out_list->det[out_list->num_detections++]=(detection_t *)block_reference(tdet);
    }

    for(int i=0;i<num_tracked;i++)
    {
        utdet_t *tdet=tracked[i];
        FILE_TRACE("FINAL tracked %2d ID=%lx [%.4f %.4f %.4f %.4f] MSS %d OBS %d TSS %5.3f TS %d",
            i, tdet->det.track_id, tdet->det.x0,tdet->det.y0,tdet->det.x1,tdet->det.y1,
            tdet->num_missed,tdet->observations, rtp_time-tdet->last_detect_time, tdet->track_state);
    }

    //detection_list_show(out_list);
    return out_list;
}