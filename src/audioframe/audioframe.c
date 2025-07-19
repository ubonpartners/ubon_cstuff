#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mutex>
#include <float.h>  // for FLT_MIN
#include "assert.h"
#include "memory_stuff.h"
#include "match.h"
#include "log.h"
#include "misc.h"
#include "audioframe.h"

struct audioframe
{
    int sample_rate;
    int num_samples;
    int num_channels;
    float samples[1];
};

static std::once_flag initFlag;
static block_allocator_t *audioframe_allocator;

static void audioframe_init()
{
    audioframe_allocator=block_allocator_create("audioframe", sizeof(audioframe_t)+160*sizeof(float));
}

audioframe_t *audioframe_create(int num_samples, int sample_rate, int num_channels)
{
    std::call_once(initFlag, audioframe_init);

    int sz=sizeof(audioframe_t)+num_samples*num_channels*sizeof(float);
    audioframe_t *fr=(audioframe_t *)block_alloc(audioframe_allocator, sz);
    if (!fr) return 0;
    memset(fr, 0, sz);
    fr->sample_rate=sample_rate;
    fr->num_samples=num_samples;
    fr->num_channels=num_channels;
    return fr;
}

audioframe_t *audioframe_reference(audioframe_t *fr)
{
    return (audioframe_t *)block_reference(fr);
}
void audioframe_destroy(audioframe_t *fr)
{
    block_free(fr);
}

float *audioframe_get_data(audioframe_t *fr)
{
    return &fr->samples[0];
}

int audioframe_get_sample_rate(audioframe_t *fr)
{
    return fr->sample_rate;
}

int audioframe_get_num_samples(audioframe_t *fr)
{
    return fr->num_samples;
}

int audioframe_get_num_channels(audioframe_t *fr)
{
    return fr->num_channels;
}

float audioframe_compute_peak(audioframe_t *fr) {
    if (!fr) return 0.f;
    int n = audioframe_get_num_samples(fr);
    int ch = audioframe_get_num_channels(fr);
    const float *d = audioframe_get_data(fr);
    float peak = 0.f;
    int total = n * ch;
    for (int i=0;i<total;i++) {
        float a = fabsf(d[i]);
        if (a > peak) peak = a;
    }
    return peak;
}

float audioframe_compute_energy(audioframe_t *fr) {
    if (!fr) return 0.f;
    int n = audioframe_get_num_samples(fr);
    int ch = audioframe_get_num_channels(fr);
    const float *d = audioframe_get_data(fr);
    double sum = 0.0; // use double for accumulator precision
    int total = n * ch;
    for (int i=0;i<total;i++) {
        float v = d[i];
        sum += (double)v * (double)v;
    }
    return (float)sum; // energy (not divided by sample count)
}

float audioframe_compute_rms(audioframe_t *fr) {
    if (!fr) return 0.f;
    int n = audioframe_get_num_samples(fr);
    int ch = audioframe_get_num_channels(fr);
    int total = n * ch;
    if (total <= 0) return 0.f;
    double sum = (double)audioframe_compute_energy(fr); // reuse above (already sums squares)
    double mean = sum / (double)total;
    return (float)sqrt(mean);
}

// Very rough LUFS approximation (NO K-weighting / gating):
// LUFS ~= -0.691 + 10 * log10(mean_square)  (constant chosen so that
// a full-scale sine ~ -3.01 dBFS maps near -3 LUFS in this simplistic model)
// This is *not* standard-compliant. Use only for relative comparisons.
float audioframe_compute_lufs_approx(audioframe_t *fr) {
    if (!fr) return 0.f;
    int n = audioframe_get_num_samples(fr);
    int ch = audioframe_get_num_channels(fr);
    int total = n * ch;
    if (total <= 0) return 0.f;
    double energy = (double)audioframe_compute_energy(fr);
    double mean_square = energy / (double)total;
    if (mean_square <= 0.0) return -90.f; // effectively silence
    double lufs = -0.691 + 10.0 * log10(mean_square);
    return (float)lufs;
}
