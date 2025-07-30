#ifndef __FAST_HISTOGRAM_H
#define __FAST_HISTOGRAM_H

#define FH_NUM_BINS     40

#include <yaml-cpp/yaml.h>

typedef struct {
    int num_bins;
    float min_val, max_val;
    // for uniform bins (bin_exp==1):
    float bin_width;
    float bin_width_inv;

    // exponential binning parameters:
    float bin_exp;       // ratio between successive bin sizes
    float log_bin_exp;   // precomputed log(bin_exp)
    float denom;         // = pow(bin_exp, num_bins) - 1
    float range_inv;     // 1/(max_val - min_val)

    // decay parameters
    float decay_factor;
    int   decay_interval;
    float decay_factor_K;
    int   sample_counter;
    uint32_t total_samples;

    float running_mean;
    float rm_alpha;

    float total;

    float bins[FH_NUM_BINS];
} fast_histogram_t;

// bin_exp defaults to 1.0f (uniform bins)
void fast_histogram_init(fast_histogram_t *h,
                         float min_val = 0.0f,
                         float max_val = 1.0f,
                         float bin_exp = 1.0f,
                         int half_life = 50,
                         int decay_interval = 10);

void fast_histogram_add_sample(fast_histogram_t *h, float v);
float fast_histogram_quantile(fast_histogram_t *h, float p);
YAML::Node fast_histogram_get_stats(fast_histogram_t *h);

#endif