#ifndef __MATHS_STUFF_H
#define __MATHS_STUFF_H

void vec_accum(float *x, float *accum, int len);
void vec_scale(float *x, float sf, int len);
void vec_mean_normalize(float *src, float *mean, float *dest, int len);
float vec_dot(float *srca, float *srcb, int len);

#endif