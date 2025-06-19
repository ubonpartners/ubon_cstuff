#ifndef __ROI_H
#define __ROI_H

typedef struct roi
{
    float box[4]; // tlx,tly,blx,bly, normalized
} roi_t;

static inline float roi_area(roi_t *r)
{
    return (r->box[2]-r->box[0])*(r->box[3]-r->box[1]);
}

static const roi_t ROI_ZERO={0};
static const roi_t ROI_ONE = { .box = {0.0f, 0.0f, 1.0f, 1.0f} };

#endif