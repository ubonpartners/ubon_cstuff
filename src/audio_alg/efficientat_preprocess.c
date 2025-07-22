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

/* ================ preprocess_cpu.c ================= */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <fftw3.h>

#include "efficientat_preprocess.h"
#include "efficientat_mel_filter.h"

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

image_t *effat_cpu_preprocess(EffatCPUHandle* h, const float* audio, int64_t n_samples){
    if(!h || !audio || n_samples <= 2) return 0;
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
        float* dest = h->mel_out;
        const float* filt = h->mel_filter + (size_t)m * bins;
        for(int f=0;f<frames;++f){
            const float* p = h->power + (size_t)f * bins;
            float acc=0.0f; for(int b=0;b<bins;++b) acc += filt[b]*p[b];
            dest[m*frames+f] = acc;
            //dest[f*c->n_mels+m]=acc;
        }
    }
    // Log + norm

    size_t total = (size_t)c->n_mels * (size_t)(frames-1);
    image_t *out=create_image_tensor(1, 1, c->n_mels, frames-1, IMAGE_FORMAT_TENSOR_FP32_HOST);
    float* out32 = (float *)out->host_mem;
    for(size_t i=0;i<total;++i) out32[i] = 0.2f * logf(h->mel_out[i] + c->eps) + 0.9f;
    free(pre); free(padded_buf);
    return out;
}
