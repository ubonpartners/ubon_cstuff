#ifndef __TRACK_AUX_H
#define __TRACK_AUX_H

#include "detections.h"

typedef struct track_aux track_aux_t;

track_aux_t *track_aux_create(track_shared_state_t *tss);
void track_aux_destroy(track_aux_t *ta);
void track_aux_run(track_aux_t *ta, image_t *img, detection_list_t *dets, bool single_frame);
void track_aux_enable_face_embeddings(track_aux_t *ta, bool enabled, float min_quality);
#endif
