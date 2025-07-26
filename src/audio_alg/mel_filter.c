#include <math.h>
#include <stdlib.h>
#include "mel_filter.h"

/* Librosa/torchlibrosa hz<->mel (HTK) */
static float hz_to_mel(float hz){ return 2595.0f * log10f(1.0f + hz/700.0f); }
static float mel_to_hz(float mel){ return 700.0f * (powf(10.0f, mel/2595.0f) - 1.0f); }

float* build_mel_filter(int sample_rate,
                        int n_fft,
                        int n_mels,
                        float fmin,
                        float fmax,
                        int norm_slaney)
{
    int n_fft_bins = n_fft/2 + 1;
    float *fb = (float*)calloc((size_t)n_mels * (size_t)n_fft_bins, sizeof(float));
    if(!fb) return NULL;

    if(fmax <= 0.0f || fmax > sample_rate/2.0f) fmax = sample_rate/2.0f;

    float mel_min  = hz_to_mel(fmin);
    float mel_max  = hz_to_mel(fmax);
    float mel_step = (mel_max - mel_min) / (n_mels + 1);

    float *mel_pts = (float*)malloc((size_t)(n_mels+2) * sizeof(float));
    float *hz_pts  = (float*)malloc((size_t)(n_mels+2) * sizeof(float));
    for(int i=0;i<n_mels+2;++i){
        mel_pts[i] = mel_min + mel_step * i;
        hz_pts[i]  = mel_to_hz(mel_pts[i]);
    }

    float *freqs = (float*)malloc((size_t)n_fft_bins * sizeof(float));
    for(int i=0;i<n_fft_bins;++i)
        freqs[i] = (sample_rate / 2.0f) * (float)i / (float)(n_fft_bins - 1);

    for(int m=0; m<n_mels; ++m){
        float f_l = hz_pts[m];
        float f_c = hz_pts[m+1];
        float f_r = hz_pts[m+2];

        for(int k=0; k<n_fft_bins; ++k){
            float f = freqs[k];
            float w = 0.0f;
            if(f >= f_l && f < f_c){
                w = (f - f_l) / (f_c - f_l);
            } else if(f >= f_c && f <= f_r){
                w = (f_r - f) / (f_r - f_c);
            }
            fb[m*n_fft_bins + k] = w;
        }

        if(norm_slaney){
            /* Slaney area normalization */
            float enorm = 2.0f / (f_r - f_l);
            for(int k=0; k<n_fft_bins; ++k)
                fb[m*n_fft_bins + k] *= enorm;
        } else {
            /* Sum-to-one normalization (approx Kaldi/EffAT) */
            float sum=0.0f;
            for(int k=0;k<n_fft_bins;++k) sum += fb[m*n_fft_bins + k];
            if(sum>0){
                float inv = 1.0f / sum;
                for(int k=0;k<n_fft_bins;++k) fb[m*n_fft_bins + k] *= inv;
            }
        }
    }

    free(mel_pts); free(hz_pts); free(freqs);
    return fb;
}
