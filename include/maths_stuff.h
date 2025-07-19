#ifndef __MATHS_STUFF_H
#define __MATHS_STUFF_H

void vec_accum(float *x, float *accum, int len);
void vec_scale(float *x, float sf, int len);
void vec_mean_normalize(float *src, float *mean, float *dest, int len);
float vec_dot(float *srca, float *srcb, int len);
void vec_l2_norm_inplace(float *p, int len);
void vec_copy_float_to_half(void *src, void *dst, int n);
void vec_copy_half_to_float(void *src, void *dst, int n);
void vec_copy_floathalf(void *src, void *dst, int n, bool src_fp16, bool dst_fp16);

#endif