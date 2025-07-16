#ifndef __UTRACK_H
#define __UTRACK_H

#include "detections.h"
#include "motion_track.h"

typedef struct utrack utrack_t;
utrack_t *utrack_create(const char *config_yaml);
void utrack_destroy(utrack_t *ut);
void utrack_reset(utrack_t *ut);
roi_t utrack_predict_positions(utrack_t *ut, double time, motion_track_t *mt, roi_t motion_roi);
detection_list_t *utrack_run(utrack_t *ut, detection_list_t *dets, double rtp_time, bool single_frame);

#endif