#ifndef EFFAT_PREPROCESSING_H
#define EFFAT_PREPROCESSING_H

#include <stdint.h>
#include "image.h"

typedef struct {
    int sample_rate;    // e.g. 32000
    int n_mels;         // e.g. 128
    int n_fft;          // e.g. 1024
    int win_length;     // e.g. 800
    int hop_length;     // e.g. 320
    float fmin;         // e.g. 0.0f
    float fmax;         // e.g. 14000.0f (or sr/2 - margin)
    float preemph;      // 0.97f
    float eps;          // 1e-5
} PreprocConfig;

/* Forward declare opaque handles */
typedef struct EffatCPUHandle EffatCPUHandle;
typedef struct EffatCUDAHandle EffatCUDAHandle;

/* Create / Destroy */
EffatCPUHandle*  effat_cpu_create(const PreprocConfig* cfg);
void             effat_cpu_destroy(EffatCPUHandle* h);
image_t *        effat_cpu_preprocess(EffatCPUHandle* h, const float* audio, int64_t n_samples);


#endif /* EFFAT_PREPROCESSING_H */