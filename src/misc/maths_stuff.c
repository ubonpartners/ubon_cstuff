#include <math.h>
#include <cuda_fp16.h>

void vec_accum(float *x, float *accum, int len)
{
    for(int i=0;i<len;i++) accum[i]+=x[i];
}

void vec_scale(float *x, float sf, int len)
{
    for(int i=0;i<len;i++) x[i]*=sf;
}

void vec_mean_normalize(float *src, float *mean, float *dest, int len)
{
    float norm=0;
    for(int i=0;i<len;i++)
    {
        float v=dest[i]=src[i]-mean[i];
        norm+=v*v;
    }
    norm=sqrtf(norm);
    if (norm!=0) vec_scale(dest, 1.0/norm, len);
}

float vec_dot(float *srca, float *srcb, int len)
{
    float ret=0;
    for(int i=0;i<len;i++) ret+=(srca[i]*srcb[i]);
    return ret;
}

void vec_l2_norm_inplace(float *p, int len)
{
    float t=0;
    for(int i=0;i<len;i++) t+=(p[i]*p[i]);
    if (t==0) return;
    float s=1.0/(sqrtf(t));
    for(int i=0;i<len;i++) p[i]*=s;
}

void vec_copy_float_to_half(void *src, void *dst, int n)
{
    __half* d = (__half*)dst;
    float *s=(float *)src;
    for (int j=0;j<n;j++) d[j]=__float2half(s[j]);
}

void vec_copy_half_to_float(void *src, void *dst, int n)
{
    __half* s = (__half*)src;
    float *d=(float *)dst;
    for (int j=0;j<n;j++) d[j]=__half2float(s[j]);
}

void vec_copy_floathalf(void *src, void *dst, int n, bool src_fp16, bool dst_fp16)
{
    if ((src_fp16==true)&&(dst_fp16==false))
    {
        vec_copy_half_to_float(src, dst, n);
    }
    else if  ((src_fp16==false)&&(dst_fp16==true))
    {
        vec_copy_float_to_half(src, dst, n);
    }
    else
    {
        memcpy(dst, src, n*(src_fp16 ? 2 : 4));
    }
}