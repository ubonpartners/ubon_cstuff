/*
 * EfficientAT Preprocessing (CPU + CUDA) Reference Implementation
 * ---------------------------------------------------------------
 * Steps (inference mode):
 *   1. Pre-emphasis y[t] = x[t] - preemph * x[t-1]  (outputs L-1 samples)
 *   2. Center reflect padding of size pad = n_fft/2 on both sides
 *   3. Framing + Hann window (periodic = false) of win_length
 *   4. Zero pad each frame tail to n_fft (if win_length < n_fft)
 *   5. Batched R2C FFT (FFTW on CPU, cuFFT on GPU)
 *   6. Power spectrum (re^2 + im^2)
 *   7. Mel projection: (n_mels x (n_fft/2+1)) * ( (n_fft/2+1) x n_frames )
 *   8. Log + affine normalization: out = 0.2 * log(mel + eps) + 0.9
 *   9. (Optional) Cast to FP16 for TensorRT input
 *
 * Accuracy goal: Numerically close (not bit-exact) to original PyTorch pipeline.
 *
 * Public C ABI (same for CPU and CUDA versions):
 *   - PreprocConfig: describes preprocessing parameters
 *   - FeatureTensor: output description
 *   - create/destroy functions for handles
 *   - preprocess_*_process for one-shot processing (re-usable plans/buffers)
 *
 * Build (CPU):
 *   gcc -O3 -fopenmp -I. -c mel_filter.c
 *   gcc -O3 -fopenmp -I. -c preprocess_cpu.c -lfftw3f -lm
 *   gcc -shared -o libeffat_pre_cpu.so preprocess_cpu.o mel_filter.o -lfftw3f -lm -fopenmp
 *
 * Build (CUDA):
 *   nvcc -O3 --compiler-options '-fPIC' -c preprocess_cuda.cu -lcufft -lcublas
 *   gcc  -O3 -fPIC -c mel_filter.c
 *   nvcc -shared -o libeffat_pre_cuda.so preprocess_cuda.o mel_filter.o -lcufft -lcublas
 *
 * NOTE: For brevity error handling is simplified. In production, add robust checks.
 */

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

/* =====================================================
 * mel_filter.h / mel_filter.c : Mel filter construction
 * Simplified Kaldi-like triangular mel filterbank.
 * ===================================================== */

#ifndef EFFAT_MEL_FILTER_H
#define EFFAT_MEL_FILTER_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Allocate and build mel filterbank matrix of shape (n_mels, n_fft_bins) in row-major. Returns pointer (malloc'd). */
float* effat_build_mel_filter(int sample_rate, int n_fft, int n_mels, float fmin, float fmax);

#ifdef __cplusplus
}
#endif
#endif

/* ================= mel_filter.c ================= */
#ifdef EFFAT_MEL_FILTER_IMPLEMENTATION
#include <math.h>
#include <stdlib.h>

static float hz_to_mel(float hz){ return 1127.0f * logf(1.0f + hz/700.0f); }
static float mel_to_hz(float mel){ return 700.0f * (expf(mel/1127.0f) - 1.0f); }

float* effat_build_mel_filter(int sample_rate, int n_fft, int n_mels, float fmin, float fmax){
    int n_fft_bins = n_fft/2 + 1; // standard R2C bins
    float *fb = (float*)calloc((size_t)n_mels * (size_t)n_fft_bins, sizeof(float));
    if(!fb) return NULL;
    if(fmax <= 0.0f || fmax > sample_rate/2.0f) fmax = sample_rate/2.0f;
    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);
    float mel_step = (mel_max - mel_min) / (n_mels + 1);
    float *mel_centers = (float*)malloc((size_t)(n_mels + 2)*sizeof(float));
    for(int i=0;i<n_mels+2;++i) mel_centers[i] = mel_min + mel_step * i;
    // convert to hz then bin index
    float *hz_centers = (float*)malloc((size_t)(n_mels + 2)*sizeof(float));
    int   *bin_centers= (int*)malloc((size_t)(n_mels + 2)*sizeof(int));
    for(int i=0;i<n_mels+2;++i){
        hz_centers[i] = mel_to_hz(mel_centers[i]);
        float bin = (n_fft * hz_centers[i]) / sample_rate; // map to FFT bin float
        bin_centers[i] = (int)floorf(bin + 0.5f);
        if(bin_centers[i] < 0) bin_centers[i]=0;
        if(bin_centers[i] > n_fft_bins-1) bin_centers[i]=n_fft_bins-1;
    }
    for(int m=0; m<n_mels; ++m){
        int left = bin_centers[m];
        int center = bin_centers[m+1];
        int right = bin_centers[m+2];
        if(center==left) center = left+1;
        if(right==center) right = center+1;
        if(right>n_fft_bins-1) right = n_fft_bins-1;
        for(int k=left;k<center;++k){
            float w = (float)(k - left) / (float)(center - left);
            fb[m*n_fft_bins + k] = w;
        }
        for(int k=center;k<=right;++k){
            float w = (float)(right - k) / (float)(right - center + 1e-12f);
            if(k < n_fft_bins) fb[m*n_fft_bins + k] = w;
        }
        // optional energy normalization (approximate Kaldi): scale so sum=1
        float sum=0.0f;
        for(int k=left;k<=right && k<n_fft_bins;++k) sum += fb[m*n_fft_bins + k];
        if(sum>0){
            float inv = 1.0f / sum;
            for(int k=left;k<=right && k<n_fft_bins;++k) fb[m*n_fft_bins + k]*=inv;
        }
    }
    free(mel_centers); free(hz_centers); free(bin_centers);
    return fb;
}
#endif /* EFFAT_MEL_FILTER_IMPLEMENTATION */

/* ================ preprocess_cpu.c ================= */
#ifdef EFFAT_PREPROCESS_CPU_IMPLEMENTATION
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <fftw3.h>

struct EffatCPUHandle {
    PreprocConfig cfg;
    float *mel_filter;      // (n_mels x (n_fft/2+1))
    float *window;          // win_length
    fftwf_plan plan;        // batched plan (created per run due to variable frames, or cached at max)
    int allocated_frames;   // capacity
    float *frames_time;     // frames * n_fft (zero-padded)
    fftwf_complex *fft_out; // frames * (n_fft/2+1)
    float *power;           // frames * (n_fft/2+1)
    float *mel_out;         // n_mels * frames
};

static float* build_hann(int win){
    float *w = (float*)malloc(sizeof(float)* (size_t)win);
    for(int i=0;i<win;++i){ w[i] = 0.5f - 0.5f * cosf( (2.0f*M_PI*i)/(win-1) ); }
    return w;
}

static void reflect_pad(const float* in, int64_t L, int pad, float* out){
    // out length = L + 2*pad
    // center pad mirrors PyTorch reflect: index mapping i<0 -> -i, i>=L -> 2L-2 - i
    for(int i=0;i<pad;++i){
        int src = pad - i; // i from left edge
        if(src >= (int)L) src = (int)L-1;
        out[i] = in[src];
    }
    memcpy(out+pad, in, sizeof(float)* (size_t)L);
    for(int i=0;i<pad;++i){
        int src = (int)L - 2 - i;
        if(src < 0) src = 0;
        out[pad + L + i] = in[src];
    }
}

EffatCPUHandle* effat_cpu_create(const PreprocConfig* cfg){
    EffatCPUHandle* h = (EffatCPUHandle*)calloc(1,sizeof(EffatCPUHandle));
    if(!h) return NULL;
    h->cfg = *cfg;
    int n_fft_bins = cfg->n_fft/2 + 1;
    h->mel_filter = effat_build_mel_filter(cfg->sample_rate, cfg->n_fft, cfg->n_mels, cfg->fmin, cfg->fmax);
    h->window = build_hann(cfg->win_length);
    h->allocated_frames = 0;
    h->frames_time = NULL; h->fft_out = NULL; h->power = NULL; h->mel_out = NULL; h->plan = NULL;
    return h;
}

void effat_cpu_destroy(EffatCPUHandle* h){
    if(!h) return;
    if(h->plan) fftwf_destroy_plan(h->plan);
    free(h->mel_filter); free(h->window); free(h->frames_time); free(h->fft_out); free(h->power); free(h->mel_out); free(h);
}

static void ensure_capacity(EffatCPUHandle* h, int frames){
    if(frames <= h->allocated_frames) return;
    int n_fft = h->cfg.n_fft;
    int bins = n_fft/2 + 1;
    size_t fcap = (size_t)frames;
    h->frames_time = (float*)realloc(h->frames_time, fcap * (size_t)n_fft * sizeof(float));
    h->fft_out = (fftwf_complex*)realloc(h->fft_out, fcap * (size_t)bins * sizeof(fftwf_complex));
    h->power = (float*)realloc(h->power, fcap * (size_t)bins * sizeof(float));
    h->mel_out = (float*)realloc(h->mel_out, (size_t)h->cfg.n_mels * fcap * sizeof(float));
    h->allocated_frames = frames;
    if(h->plan) fftwf_destroy_plan(h->plan);
    h->plan = fftwf_plan_many_dft_r2c(1, &n_fft, frames,
              h->frames_time, NULL, 1, n_fft,
              h->fft_out, NULL, 1, (n_fft/2+1), FFTW_MEASURE);
}

int effat_cpu_preprocess(EffatCPUHandle* h, const float* audio, int64_t n_samples, FeatureTensor* out){
    if(!h || !audio || n_samples <= 2) return -1;
    const PreprocConfig* c = &h->cfg;
    int64_t Lp = n_samples - 1; // after pre-emphasis conv1d valid
    int pad = c->n_fft/2;
    int64_t padded = Lp + 2*pad;
    int frames = (int)((padded - c->win_length) / c->hop_length + 1);
    if(frames < 1) frames = 1;
    ensure_capacity(h, frames);
    // Pre-emphasis into temp buffer (reuse frames_time area tail). We'll allocate temp arrays.
    float* pre = (float*)malloc(sizeof(float)*(size_t)Lp);
    for(int64_t t=0; t<Lp; ++t){
        float x_t = audio[t+1];
        float x_prev = audio[t];
        pre[t] = x_t - c->preemph * x_prev;
    }
    // Reflect pad
    float* padded_buf = (float*)malloc(sizeof(float)*(size_t)(padded));
    reflect_pad(pre, Lp, pad, padded_buf);
    // Framing + window + zero pad
    for(int f=0; f<frames; ++f){
        int64_t start = (int64_t)f * c->hop_length;
        float* frame = h->frames_time + (size_t)f * c->n_fft;
        for(int i=0;i<c->win_length;++i){
            int64_t idx = start + i;
            float sample = 0.0f;
            if(idx >=0 && idx < padded) sample = padded_buf[idx];
            frame[i] = sample * h->window[i];
        }
        for(int i=c->win_length;i<c->n_fft;++i) frame[i]=0.0f;
    }
    // FFT
    fftwf_execute(h->plan);
    int bins = c->n_fft/2 + 1;
    // Power
    for(int f=0; f<frames; ++f){
        fftwf_complex* row = h->fft_out + (size_t)f * bins;
        float* prow = h->power + (size_t)f * bins;
        for(int b=0;b<bins;++b){ float re=row[b][0]; float im=row[b][1]; prow[b]=re*re+im*im; }
    }
    // Mel projection: mel_out (n_mels x frames)
    for(int m=0;m<c->n_mels;++m){
        float* dest = h->mel_out + (size_t)m * frames;
        const float* filt = h->mel_filter + (size_t)m * bins;
        for(int f=0;f<frames;++f){
            const float* p = h->power + (size_t)f * bins;
            float acc=0.0f; for(int b=0;b<bins;++b) acc += filt[b]*p[b];
            dest[f] = acc;
        }
    }
    // Log + norm & (optional) FP16 cast
    size_t total = (size_t)c->n_mels * (size_t)frames;
    if(c->cast_to_fp16){
        uint16_t* out16 = (uint16_t*)malloc(total * sizeof(uint16_t));
        for(size_t i=0;i<total;++i){
            float v = 0.2f * logf(h->mel_out[i] + c->eps) + 0.9f;
            // float32 -> fp16 (naive) IEEE 754 rounding (simple conversion)
            union { float f; uint32_t u; } pun; pun.f = v;
            uint32_t u = pun.u;
            uint16_t sign = (u >> 16) & 0x8000u;
            int exp = ((u >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = u & 0x7FFFFFu;
            uint16_t half;
            if(exp <= 0){ half = sign; }
            else if(exp >= 31){ half = sign | 0x7C00u; }
            else { half = sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13); }
            out16[i] = half;
        }
        out->data = out16;
        out->dtype = EFFAT_DTYPE_F16;
    } else {
        float* out32 = (float*)malloc(total * sizeof(float));
        for(size_t i=0;i<total;++i) out32[i] = 0.2f * logf(h->mel_out[i] + c->eps) + 0.9f;
        out->data = out32;
        out->dtype = EFFAT_DTYPE_F32;
    }
    out->n_mels = c->n_mels;
    out->n_frames = frames;
    free(pre); free(padded_buf);
    return 0;
}

void effat_release_tensor(const FeatureTensor* out, int device_is_cuda){
    if(!out || device_is_cuda) return; // CPU only here
    if(out->data) free(out->data);
}

#endif /* EFFAT_PREPROCESS_CPU_IMPLEMENTATION */

/* ================= preprocess_cuda.cu ================= */
#ifdef EFFAT_PREPROCESS_CUDA_IMPLEMENTATION
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ fprintf(stderr,"CUDA Error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); return -1;} } while(0)
#endif
#ifndef CHECK_CUFFT
#define CHECK_CUFFT(x) do { cufftResult err=(x); if(err!=CUFFT_SUCCESS){ fprintf(stderr,"cuFFT Error %s:%d code %d\n",__FILE__,__LINE__,err); return -1;} } while(0)
#endif
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(x) do { cublasStatus_t err=(x); if(err!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"cuBLAS Error %s:%d code %d\n",__FILE__,__LINE__,err); return -1;} } while(0)
#endif

struct EffatCUDAHandle {
    PreprocConfig cfg;
    float *d_mel_filter; // (n_mels x bins)
    float *d_window;     // win_length
    // dynamic buffers sized to max processed length so far
    size_t cap_samples;  // capacity for raw audio
    float *d_pre;        // (L-1)
    float *d_padded;     // (L-1 + 2*pad)
    float *d_frames;     // frames * n_fft (float)
    cufftComplex *d_fft; // frames * bins
    float *d_power;      // frames * bins
    float *d_mel_out;    // mels * frames (float)
    void *d_out;         // final output (float or half)
    int allocated_frames;
    cufftHandle plan;
    cublasHandle_t cublas;
};

static __global__ void k_preemph(const float* __restrict__ x, float* __restrict__ y, int64_t L, float a){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < L-1){
        float x1 = x[idx+1];
        float x0 = x[idx];
        y[idx] = x1 - a * x0;
    }
}

static __global__ void k_reflect_pad(const float* __restrict__ in, int64_t Lin, int pad, float* __restrict__ out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t Lout = Lin + 2*pad;
    if(idx < Lout){
        int64_t j = idx - pad;
        int64_t Lm1 = Lin - 1;
        if(j < 0) j = -j; // reflect
        if(j > Lm1) j = 2*Lm1 - j;
        if(j < 0) j=0; if(j> Lm1) j=Lm1;
        out[idx] = in[j];
    }
}

static __global__ void k_frame_window(const float* __restrict__ padded, int64_t Lpad, int win, int hop, int n_fft, int frames, const float* __restrict__ window, float* __restrict__ frames_out){
    int f = blockIdx.x;
    int i = threadIdx.x;
    if(f < frames && i < n_fft){
        float v = 0.0f;
        if(i < win){
            int64_t start = (int64_t)f * hop;
            int64_t idx = start + i;
            if(idx >=0 && idx < Lpad) v = padded[idx] * window[i];
        }
        frames_out[f * n_fft + i] = v;
    }
}

static __global__ void k_power(const cufftComplex* __restrict__ fft, float* __restrict__ power, int bins, int frames){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = frames * bins;
    if(idx < total){
        cufftComplex z = fft[idx];
        power[idx] = z.x * z.x + z.y * z.y;
    }
}

static __global__ void k_log_norm_cast(const float* __restrict__ mel, int mels, int frames, float eps, int cast_fp16, void* __restrict__ out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = mels * frames;
    if(idx < total){
        float v = mel[idx];
        v = 0.2f * logf(v + eps) + 0.9f;
        if(cast_fp16){
            // simple float->half (round to nearest) using CUDA intrinsic
            __half hv = __float2half(v);
            ((__half*)out)[idx] = hv;
        } else {
            ((float*)out)[idx] = v;
        }
    }
}

static float* device_alloc_or_resize_float(float** ptr, size_t needed){
    if(*ptr){ size_t current; cudaPointerAttributes attr; cudaPointerGetAttributes(&attr, *ptr); /* not reliable for size */ }
    // naive: free & reallocate
    if(*ptr){ cudaFree(*ptr); *ptr=NULL; }
    cudaMalloc((void**)ptr, needed * sizeof(float));
    return *ptr;
}

EffatCUDAHandle* effat_cuda_create(const PreprocConfig* cfg){
    EffatCUDAHandle* h = (EffatCUDAHandle*)calloc(1,sizeof(EffatCUDAHandle));
    if(!h) return NULL;
    h->cfg = *cfg; h->cap_samples = 0; h->allocated_frames = 0; h->plan = 0;
    int bins = cfg->n_fft/2 + 1;
    // Host build mel + window then upload
    float* mel = effat_build_mel_filter(cfg->sample_rate, cfg->n_fft, cfg->n_mels, cfg->fmin, cfg->fmax);
    float* win = (float*)malloc(sizeof(float)* (size_t)cfg->win_length);
    for(int i=0;i<cfg->win_length;++i) win[i] = 0.5f - 0.5f * cosf((2.0f*M_PI*i)/(cfg->win_length-1));
    cudaMalloc((void**)&h->d_mel_filter, (size_t)cfg->n_mels * bins * sizeof(float));
    cudaMemcpy(h->d_mel_filter, mel, (size_t)cfg->n_mels * bins * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&h->d_window, (size_t)cfg->win_length * sizeof(float));
    cudaMemcpy(h->d_window, win, (size_t)cfg->win_length * sizeof(float), cudaMemcpyHostToDevice);
    free(mel); free(win);
    cublasCreate(&h->cublas);
    return h;
}

void effat_cuda_destroy(EffatCUDAHandle* h){
    if(!h) return;
    if(h->plan) cufftDestroy(h->plan);
    cublasDestroy(h->cublas);
    cudaFree(h->d_mel_filter); cudaFree(h->d_window);
    cudaFree(h->d_pre); cudaFree(h->d_padded); cudaFree(h->d_frames);
    cudaFree(h->d_fft); cudaFree(h->d_power); cudaFree(h->d_mel_out); cudaFree(h->d_out);
    free(h);
}

static int ensure_cuda_capacity(EffatCUDAHandle* h, int64_t n_samples, int frames){
    const PreprocConfig* c = &h->cfg;
    int bins = c->n_fft/2 + 1;
    size_t need_pre = (size_t)(n_samples-1);
    size_t need_pad = need_pre + (size_t)c->n_fft; // approximate upper bound (2*pad = n_fft)
    size_t need_frames = (size_t)frames * c->n_fft;
    size_t need_fft = (size_t)frames * bins;
    if(need_pre > h->cap_samples){
        if(h->d_pre) {cudaFree(h->d_pre); h->d_pre=NULL;}
        cudaMalloc((void**)&h->d_pre, need_pre * sizeof(float));
        if(h->d_padded) {cudaFree(h->d_padded); h->d_padded=NULL;}
        cudaMalloc((void**)&h->d_padded, need_pad * sizeof(float));
        h->cap_samples = need_pre;
    }
    if(frames > h->allocated_frames){
        if(h->d_frames) cudaFree(h->d_frames);
        cudaMalloc((void**)&h->d_frames, need_frames * sizeof(float));
        if(h->d_fft) cudaFree(h->d_fft);
        cudaMalloc((void**)&h->d_fft, need_fft * sizeof(cufftComplex));
        if(h->d_power) cudaFree(h->d_power);
        cudaMalloc((void**)&h->d_power, need_fft * sizeof(float));
        if(h->d_mel_out) cudaFree(h->d_mel_out);
        cudaMalloc((void**)&h->d_mel_out, (size_t)h->cfg.n_mels * frames * sizeof(float));
        if(h->d_out) {cudaFree(h->d_out); h->d_out=NULL;}
        if(h->cfg.cast_to_fp16) cudaMalloc(&h->d_out, (size_t)h->cfg.n_mels * frames * sizeof(__half));
        else cudaMalloc(&h->d_out, (size_t)h->cfg.n_mels * frames * sizeof(float));
        // Rebuild cuFFT plan
        if(h->plan) cufftDestroy(h->plan);
        int n[1] = { h->cfg.n_fft };
        CHECK_CUFFT(cufftPlanMany(&h->plan, 1, n,
            NULL, 1, h->cfg.n_fft,
            NULL, 1, (h->cfg.n_fft/2+1), CUFFT_R2C, frames));
        h->allocated_frames = frames;
    }
    return 0;
}

int effat_cuda_preprocess(EffatCUDAHandle* h, const float* h_audio, int64_t n_samples, FeatureTensor* out){
    if(!h || !h_audio || n_samples < 2) return -1;
    const PreprocConfig* c = &h->cfg;
    int64_t Lp = n_samples - 1; // preemphasis output length
    int pad = c->n_fft/2;
    int64_t padded = Lp + 2*pad;
    int frames = (int)((padded - c->win_length)/c->hop_length + 1);
    if(frames < 1) frames = 1;
    ensure_cuda_capacity(h, n_samples, frames);
    // Upload raw audio (reuse d_padded as staging? allocate ephemeral?)
    float* d_audio_raw = NULL; cudaMalloc((void**)&d_audio_raw, n_samples * sizeof(float));
    cudaMemcpy(d_audio_raw, h_audio, n_samples * sizeof(float), cudaMemcpyHostToDevice);
    // Preemphasis
    int threads = 256; int blocks = (int)((Lp + threads -2)/threads);
    k_preemph<<<blocks, threads>>>(d_audio_raw, h->d_pre, n_samples, c->preemph);
    // Reflect pad
    int64_t Lout = padded;
    int blocks_pad = (int)((Lout + threads -1)/threads);
    k_reflect_pad<<<blocks_pad, threads>>>(h->d_pre, Lp, pad, h->d_padded);
    // Frames + window
    dim3 gridF(frames); dim3 blockF(c->n_fft);
    if(blockF.x > 1024) blockF.x = 1024; // safety
    k_frame_window<<<gridF, blockF>>>(h->d_padded, padded, c->win_length, c->hop_length, c->n_fft, frames, h->d_window, h->d_frames);
    // FFT (in-place: frames stored row-major contiguous). We planned for frames * n_fft.
    CHECK_CUFFT(cufftSetStream(h->plan, 0));
    cufftExecR2C(h->plan, (cufftReal*)h->d_frames, h->d_fft);
    // Power
    int bins = c->n_fft/2 + 1;
    int total_bins = frames * bins;
    int blocks_pw = (total_bins + threads -1)/threads;
    k_power<<<blocks_pw, threads>>>(h->d_fft, h->d_power, bins, frames);
    // Mel projection: mel_out (mels x frames) = mel (mels x bins) * power (bins x frames)
    float alpha=1.0f, beta=0.0f;
    // Using column-major math: treat our row-major arrays by transposing dimensions logically.
    // We'll launch cublasSgemm with operands (C = A * B): A (mels x bins), B (bins x frames), C (mels x frames)
    CHECK_CUBLAS(cublasSgemm(h->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        c->n_mels, frames, bins,
        &alpha, h->d_mel_filter, c->n_mels, h->d_power, bins, &beta, h->d_mel_out, c->n_mels));
    // Log + norm + cast
    int total = c->n_mels * frames;
    int blocks_ln = (total + threads -1)/threads;
    k_log_norm_cast<<<blocks_ln, threads>>>(h->d_mel_out, c->n_mels, frames, c->eps, c->cast_to_fp16, h->d_out);
    cudaFree(d_audio_raw);
    out->n_mels = c->n_mels;
    out->n_frames = frames;
    out->data = h->d_out;
    out->dtype = c->cast_to_fp16 ? EFFAT_DTYPE_F16 : EFFAT_DTYPE_F32;
    cudaDeviceSynchronize();
    return 0;
}

#endif /* EFFAT_PREPROCESS_CUDA_IMPLEMENTATION */