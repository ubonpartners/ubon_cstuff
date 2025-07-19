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