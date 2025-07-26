#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stdint.h>
#include "image.h"

/* Which frontend? */
typedef enum {
    PREPROC_EFFAT = 0,
    PREPROC_TINYCLAP = 1
} PreprocFlavor;

typedef struct {
    /* Common STFT/Mel params */
    int   sample_rate;      // 44100 for tinyCLAP, 32000 for EfficientAT
    int   n_mels;           // 64 tinyCLAP, 128 EfficientAT (typical)
    int   n_fft;            // 1024
    int   win_length;       // 1024 tinyCLAP, 800 EfficientAT default
    int   hop_length;       // 320
    float fmin;             // 50.0 tinyCLAP, 0.0 EfficientAT typical
    float fmax;             // 14000 tinyCLAP, sr/2 for EfficientAT typical

    /* Extra knobs to unify both recipes */
    PreprocFlavor flavor;   // choose PREPROC_TINYCLAP or PREPROC_EFFAT
    float preemph;          // 0.97 for EfficientAT, 0.0 for tinyCLAP
    int   use_preemph;      // 1/0

    /* Magnitude exponent: 1 = |X|, 2 = power. tinyCLAP uses 1, EffAT=2 */
    float spec_mag_power;

    /* Log transform: natural vs. log10? scale/shift? */
    int   log_base_e;       // 1 -> ln, 0 -> log10
    float log_mul;          // multiplier (EffAT: 0.2, tinyCLAP: 1.0)
    float log_add;          // bias (EffAT: 0.9, tinyCLAP: 0.0)
    float eps;              // 1e-10 or 1e-5

    /* Padding */
    int   center;           // center=True (torchlibrosa), 1/0
    int   pad_mode_reflect; // only reflect used here, keep for completeness

    /* Mel filter norm: 0=sum1, 1=Slaney area */
    int   mel_norm_slaney;  // tinyCLAP: 1, EfficientAT: 0

    // tensor format
    bool mels_inner_dim; // if output tensor is frames x mels (like tinyclap) vs (mels x fames, efficientat)

} PreprocConfig;

/* Opaque handles */
typedef struct audio_preproc audio_preproc_t;

/* Create / Destroy */
audio_preproc_t * audio_preproc_create(const PreprocConfig* cfg);
void              audio_preproc_destroy(audio_preproc_t* h);
image_t          *audio_preprocess(audio_preproc_t* h, const float* audio, int64_t n_samples);

/* --- Optional: handy presets --- */
void preproc_fill_tinyclap_defaults(PreprocConfig* c);
void preproc_fill_efficientat_defaults(PreprocConfig* c);

#endif /* EFFAT_PREPROCESSING_H */