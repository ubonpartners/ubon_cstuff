#ifndef EFFICIENTAT_MEL_FILTER_H
#define EFFICIENTAT_MEL_FILTER_H

float* build_mel_filter(int sample_rate,
                        int n_fft,
                        int n_mels,
                        float fmin,
                        float fmax,
                        int   norm_slaney); /* 1=Slaney, 0=sum1 */

#endif