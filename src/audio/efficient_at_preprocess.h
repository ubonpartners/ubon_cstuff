#ifndef EFFAT_PREPROCESSING_H
#define EFFAT_PREPROCESSING_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

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
    int   cast_to_fp16; // 1 -> output FP16 data
} PreprocConfig;

typedef enum { EFFAT_DTYPE_F32 = 0, EFFAT_DTYPE_F16 = 1 } EffatDType;

typedef struct {
    int n_mels;     // mel bands
    int n_frames;   // time frames
    void *data;     // pointer to contiguous data (layout: (1,1,n_mels,n_frames))
    EffatDType dtype;
} FeatureTensor;

/* Forward declare opaque handles */
typedef struct EffatCPUHandle EffatCPUHandle;
typedef struct EffatCUDAHandle EffatCUDAHandle;

/* Create / Destroy */
EffatCPUHandle*  effat_cpu_create(const PreprocConfig* cfg);
void             effat_cpu_destroy(EffatCPUHandle* h);
int              effat_cpu_preprocess(EffatCPUHandle* h, const float* audio, int64_t n_samples, FeatureTensor* out);

EffatCUDAHandle* effat_cuda_create(const PreprocConfig* cfg); // returns NULL if CUDA not available
void             effat_cuda_destroy(EffatCUDAHandle* h);
int              effat_cuda_preprocess(EffatCUDAHandle* h, const float* h_audio, int64_t n_samples, FeatureTensor* out);

/* Utility: release FeatureTensor data allocated by library (CPU path). On CUDA path data is device pointer; caller frees separately or via dedicated function. */
void effat_release_tensor(const FeatureTensor* out, int device_is_cuda);

#ifdef __cplusplus
}
#endif
#endif /* EFFAT_PREPROCESSING_H */