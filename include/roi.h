#ifndef __ROI_H
#define __ROI_H

#include <algorithm>

typedef struct roi
{
    float box[4]; // tlx,tly,blx,bly, normalized
} roi_t;

static inline float roi_area(roi_t *r)
{
    return (r->box[2]-r->box[0])*(r->box[3]-r->box[1]);
}

static inline roi_t roi_union(roi_t a, roi_t b)
{
    roi_t ret;
    ret.box[0]=std::min(a.box[0],b.box[0]);
    ret.box[1]=std::min(a.box[1],b.box[1]);
    ret.box[2]=std::max(a.box[2],b.box[2]);
    ret.box[3]=std::max(a.box[3],b.box[3]);
    return ret;
}

static const roi_t ROI_ZERO={0};
static const roi_t ROI_ONE = { .box = {0.0f, 0.0f, 1.0f, 1.0f} };

roi_t expand_roi_to_aspect_ratio(roi_t r, float a);

#endif