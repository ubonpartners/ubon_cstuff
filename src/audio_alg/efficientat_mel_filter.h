/* =====================================================
 * mel_filter.h / mel_filter.c : Mel filter construction
 * Simplified Kaldi-like triangular mel filterbank.
 * ===================================================== */
#ifndef EFFAT_MEL_FILTER_H
#define EFFAT_MEL_FILTER_H
#include <stddef.h>

/* Allocate and build mel filterbank matrix of shape (n_mels, n_fft_bins) in row-major. Returns pointer (malloc'd). */
float* effat_build_mel_filter(int sample_rate, int n_fft, int n_mels, float fmin, float fmax);

#endif