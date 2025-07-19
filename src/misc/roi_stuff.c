#include <stdint.h>
#include "roi.h"

roi_t expand_roi_to_aspect_ratio(roi_t r, float a) {
    // 1. original width/height & center
    float x0 = r.box[0], y0 = r.box[1];
    float x1 = r.box[2], y1 = r.box[3];
    float rw = x1 - x0, rh = y1 - y0;
    float cx = (x0 + x1) * 0.5, cy = (y0 + y1) * 0.5;

    // compute minimal new size
    float new_w = rw < a*rh ? a*rh : rw;
    float new_h = rh < rw/a ? rw/a : rh;

    // if either dimension exceeds 1.0, fallback to the whole unit square
    if (new_w > 1.0 || new_h > 1.0) {
        return ROI_ONE;
    }

    // center the new box
    float nx0 = cx - new_w * 0.5;
    float ny0 = cy - new_h * 0.5;
    float nx1 = cx + new_w * 0.5;
    float ny1 = cy + new_h * 0.5;

    // clamp by shifting
    if (nx0 < 0.0) { nx1 -= nx0; nx0 = 0.0; }
    if (nx1 > 1.0) { nx0 -= (nx1 - 1.0); nx1 = 1.0; }
    if (ny0 < 0.0) { ny1 -= ny0; ny0 = 0.0; }
    if (ny1 > 1.0) { ny0 -= (ny1 - 1.0); ny1 = 1.0; }

    roi_t out;
    out.box[0] = nx0;
    out.box[1] = ny0;
    out.box[2] = nx1;
    out.box[3] = ny1;
    return out;
}