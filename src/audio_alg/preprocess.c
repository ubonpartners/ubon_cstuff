/*
 * Unified CPU Preprocessing for EfficientAT / tinyCLAP
 * ----------------------------------------------------
 * Implements:
 *  - Optional pre-emphasis
 *  - center reflect padding
 *  - Hann window
 *  - |STFT|^spec_mag_power
 *  - Mel projection (Slaney or sum1)
 *  - log transform with configurable base/scale/bias
 *
 * Dependencies:
 *   FFTW3 (single precision): -lfftw3f
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include <fftw3.h>

#include "preprocess.h"
#include "mel_filter.h"

#define PI_F 3.14159265358979323846f

struct audio_preproc {
    PreprocConfig cfg;

    float *mel_filter;      // (n_mels x (n_fft/2+1))
    float *window;          // win_length

    /* Workspace */
    int   allocated_frames;
    float *frames_time;     // frames x n_fft
    fftwf_complex *fft_out; // frames x (n_fft/2+1)
    float *spec;            // frames x (n_fft/2+1)  (mag or power)
    float *mel_tmp;         // n_mels x frames (temp before log)

    fftwf_plan plan;
};

/* ---------------- Utils ---------------- */

static float* build_hann(int win){
    float *w = (float*)malloc(sizeof(float)*(size_t)win);
    for(int i=0;i<win;++i){
        w[i] = 0.5f - 0.5f * cosf((2.0f * PI_F * i) / (float)(win-1));
    }
    return w;
}

/* PyTorch-like reflect pad with center=True (pad = win_length/2) */
static void reflect_pad(const float* in, int64_t L, int pad, float* out){
    // left
    for(int i=0;i<pad;++i){
        int src = pad - i;
        if(src >= L) src = (int)L-1;
        out[i] = in[src];
    }
    // middle
    memcpy(out+pad, in, sizeof(float)*(size_t)L);
    // right
    for(int i=0;i<pad;++i){
        int src = (int)L - 2 - i;
        if(src < 0) src = 0;
        out[pad + L + i] = in[src];
    }
}

/* Capacity mgmt */
static void ensure_capacity(audio_preproc_t* h, int frames){
    if(frames <= h->allocated_frames) return;
    int n_fft = h->cfg.n_fft;
    int bins  = n_fft/2 + 1;

    size_t fcap = (size_t)frames;
    h->frames_time = (float*)realloc(h->frames_time, fcap*(size_t)n_fft*sizeof(float));
    h->fft_out     = (fftwf_complex*)realloc(h->fft_out, fcap*(size_t)bins*sizeof(fftwf_complex));
    h->spec        = (float*)realloc(h->spec, fcap*(size_t)bins*sizeof(float));
    h->mel_tmp     = (float*)realloc(h->mel_tmp, (size_t)h->cfg.n_mels*fcap*sizeof(float));
    h->allocated_frames = frames;

    if(h->plan) fftwf_destroy_plan(h->plan);
    h->plan = fftwf_plan_many_dft_r2c(
        1, &n_fft, frames,
        h->frames_time, NULL, 1, n_fft,
        h->fft_out, NULL, 1, bins,
        FFTW_MEASURE
    );
}

/* ---------------- Public API ---------------- */

audio_preproc_t* audio_preproc_create(const PreprocConfig* cfg_in){
    audio_preproc_t* h = (audio_preproc_t*)calloc(1,sizeof(audio_preproc_t));
    if(!h) return NULL;
    h->cfg = *cfg_in;

    int bins = cfg_in->n_fft/2 + 1;
    h->mel_filter = build_mel_filter(cfg_in->sample_rate,
                                     cfg_in->n_fft,
                                     cfg_in->n_mels,
                                     cfg_in->fmin,
                                     cfg_in->fmax,
                                     cfg_in->mel_norm_slaney);
    h->window = build_hann(cfg_in->win_length);

    h->allocated_frames = 0;
    h->frames_time = NULL; h->fft_out = NULL; h->spec = NULL; h->mel_tmp = NULL;
    h->plan = NULL;
    return h;
}

void audio_preproc_destroy(audio_preproc_t* h){
    if(!h) return;
    if(h->plan) fftwf_destroy_plan(h->plan);
    free(h->mel_filter);
    free(h->window);
    free(h->frames_time);
    free(h->fft_out);
    free(h->spec);
    free(h->mel_tmp);
    free(h);
}

image_t *audio_preprocess(audio_preproc_t* h,
                              const float* audio,
                              int64_t n_samples)
{
    if(!h || !audio || n_samples <= 0) return NULL;
    const PreprocConfig* c = &h->cfg;

    /* Step 0: optional pre-emphasis */
    const float* x_src = audio;
    float* x_pre = NULL;
    int64_t Lp = n_samples;
    if(c->use_preemph && fabsf(c->preemph) > 1e-9f){
        Lp = n_samples - 1;
        x_pre = (float*)malloc(sizeof(float)*(size_t)Lp);
        for(int64_t t=0; t<Lp; ++t){
            x_pre[t] = audio[t+1] - c->preemph * audio[t];
        }
        x_src = x_pre;
    }

    /* Step 1: center reflect pad */
    int pad = c->center ? (c->win_length/2) : 0;
    int64_t padded = Lp + 2*pad;
    float* x_pad = (float*)malloc(sizeof(float)*(size_t)padded);
    if(c->center){
        reflect_pad(x_src, Lp, pad, x_pad);
    } else {
        memcpy(x_pad, x_src, sizeof(float)*(size_t)Lp);
        if(padded > Lp) memset(x_pad + Lp, 0, sizeof(float)*(size_t)(padded-Lp));
    }

    if(x_pre) free(x_pre);

    /* Step 2: compute #frames */
    int64_t frames64 = 1 + (padded - c->win_length) / c->hop_length;
    if(frames64 < 1) frames64 = 1;
    int frames = (int)frames64;

    ensure_capacity(h, frames);

    /* Step 3: frame, window, zero-pad to n_fft */
    for(int f=0; f<frames; ++f){
        int64_t start = (int64_t)f * c->hop_length;
        float* frame = h->frames_time + (size_t)f * c->n_fft;

        for(int i=0;i<c->win_length;++i){
            int64_t idx = start + i;
            float sample = (idx >=0 && idx < padded) ? x_pad[idx] : 0.0f;
            frame[i] = sample * h->window[i];
        }
        for(int i=c->win_length;i<c->n_fft;++i) frame[i]=0.0f;
    }

    free(x_pad);

    /* Step 4: FFT */
    fftwf_execute(h->plan);

    int bins = c->n_fft/2 + 1;

    /* Step 5: magnitude or power */
    for(int f=0; f<frames; ++f){
        fftwf_complex* row = h->fft_out + (size_t)f * bins;
        float* srow = h->spec + (size_t)f * bins;
        for(int b=0;b<bins;++b){
            float re = row[b][0], im = row[b][1];
            float mag = sqrtf(re*re + im*im);
            if(c->spec_mag_power == 2.0f) srow[b] = mag * mag;
            else                           srow[b] = mag;      // default tinyCLAP
        }
    }

    /* Step 6: mel projection  mel_tmp[m, f] */
    for(int m=0; m<c->n_mels; ++m){
        const float* filt = h->mel_filter + (size_t)m * bins;
        for(int f=0; f<frames; ++f){
            const float* p = h->spec + (size_t)f * bins;
            float acc=0.0f;
            for(int b=0;b<bins;++b) acc += filt[b] * p[b];
            h->mel_tmp[m*frames + f] = acc;
        }
    }

    /* Step 7: log + affine -> output */
    image_t *out = create_image_tensor(1, 1, c->n_mels, frames,
                                       IMAGE_FORMAT_TENSOR_FP32_HOST);
    float *dst = (float*)out->host_mem;

    const float log_eps = (c->eps <= 0.0f) ? 1e-10f : c->eps;
    if(c->log_base_e){
        for(size_t i=0, n=(size_t)c->n_mels*frames; i<n; ++i){
            dst[i] = c->log_mul * logf(h->mel_tmp[i] + log_eps) + c->log_add;
        }
    } else {
        const float inv_ln10 = 1.0f / logf(10.0f);
        for(size_t i=0, n=(size_t)c->n_mels*frames; i<n; ++i){
            dst[i] = c->log_mul * (logf(h->mel_tmp[i] + log_eps) * inv_ln10) + c->log_add;
        }
    }

    return out;
}

/* ---------------- Presets ---------------- */

void preproc_fill_tinyclap_defaults(PreprocConfig* c){
    memset(c, 0, sizeof(*c));
    c->flavor          = PREPROC_TINYCLAP;
    c->sample_rate     = 44100;
    c->n_mels          = 64;
    c->n_fft           = 1024;
    c->win_length      = 1024;
    c->hop_length      = 320;
    c->fmin            = 50.0f;
    c->fmax            = 14000.0f;

    c->use_preemph     = 0;
    c->preemph         = 0.0f;

    c->spec_mag_power  = 1.0f;

    c->log_base_e      = 1;
    c->log_mul         = 1.0f;
    c->log_add         = 0.0f;
    c->eps             = 1e-10f;

    c->center          = 1;
    c->pad_mode_reflect= 1;

    c->mel_norm_slaney = 1;
}

void preproc_fill_efficientat_defaults(PreprocConfig* c){
    memset(c, 0, sizeof(*c));
    c->flavor          = PREPROC_EFFAT;
    c->sample_rate     = 32000;
    c->n_mels          = 128;
    c->n_fft           = 1024;
    c->win_length      = 800;
    c->hop_length      = 320;
    c->fmin            = 0.0f;
    c->fmax            = 0.0f;   // will clamp to sr/2

    c->use_preemph     = 1;
    c->preemph         = 0.97f;

    c->spec_mag_power  = 2.0f;

    c->log_base_e      = 1;      // or 0 if you used log10 internally
    c->log_mul         = 0.2f;
    c->log_add         = 0.9f;
    c->eps             = 1e-5f;

    c->center          = 1;
    c->pad_mode_reflect= 1;

    c->mel_norm_slaney = 0;
}
