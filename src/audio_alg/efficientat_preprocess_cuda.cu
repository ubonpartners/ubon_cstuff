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
