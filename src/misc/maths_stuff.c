#include <math.h>

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