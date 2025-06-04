#ifndef __ROI_H
#define __ROI_H

typedef struct roi
{
    float box[4]; // tlx,tly,blx,bly, normalized
} roi_t;

#endif