#ifndef __MOTA_METRICS_H
#define __MOTA_METRICS_H

#include "detections.h"

typedef struct metric_results
{
    int num_frames;
    int num_objects;
    int mostly_tracked;
    int partially_tracked;
    int mostly_lost;
    int num_false_positives;
    int num_misses;
    int num_switches;
    int num_fragmentations;
    int num_unique_objects;
    int num_matches;
    double total_iou;
    int missed;
    int fp_tracks;
    // Derived
    double recall;
    double precision;
    double mota;
    double motp;
    int idfp;
    int idfn;
    int idtp;
    double idp;
    double idr;
    double idf1;
} metric_results_t;

typedef struct mota_metrics mota_metrics_t;

mota_metrics_t *mota_metrics_create();
void mota_metrics_add_frame(mota_metrics_t *mm, detection_t **dets, int num_dets, detection_t **gts, int num_gts);
void mota_metrics_get_results(mota_metrics_t *mm, metric_results_t *out);
void mota_metrics_destroy(mota_metrics_t *mm);


#endif
