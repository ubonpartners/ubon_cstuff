#include "fast_histogram.h"
#include <math.h>
#include <assert.h>
#include <string.h>

void fast_histogram_init(fast_histogram_t *h,
                         float min_val,
                         float max_val,
                         float bin_exp,
                         int   half_life,
                         int   decay_interval)
{
    assert(max_val > min_val);
    assert(half_life > 0 && decay_interval > 0);
    assert(bin_exp > 0.0f);

    memset(h, 0, sizeof(*h));
    h->num_bins      = FH_NUM_BINS;
    h->min_val       = min_val;
    h->max_val       = max_val;
    h->bin_exp       = bin_exp;
    h->range_inv     = 1.0f / (max_val - min_val);

    if (bin_exp == 1.0f) {
        // uniform bins
        h->bin_width     = (max_val - min_val) / h->num_bins;
        h->bin_width_inv = h->num_bins / (max_val - min_val);
        h->log_bin_exp   = 0.0f;
        h->denom         = 0.0f;
    } else {
        // exponential bins
        h->log_bin_exp = logf(bin_exp);
        h->denom       = powf(bin_exp, h->num_bins) - 1.0f;
        // bin_width/bin_width_inv unused in this mode
    }

    // decay setup
    h->decay_factor   = powf(0.5f, 1.0f / half_life);
    h->decay_interval = decay_interval;
    h->decay_factor_K = powf(h->decay_factor, decay_interval);
    h->sample_counter = 0;
    h->total_samples  = 0;
    h->running_mean   = 0.0f;
    h->rm_alpha       = 1.0f;
}

void fast_histogram_add_sample(fast_histogram_t *h, float v)
{
    h->running_mean+=h->rm_alpha * (v-h->running_mean);
    h->rm_alpha=std::max(0.01f, h->rm_alpha*0.95f);
    h->total+=v;

    // apply decay if needed
    if (++h->sample_counter >= h->decay_interval) {
        for (int i = 0; i < h->num_bins; i++) {
            h->bins[i] *= h->decay_factor_K;
        }
        h->sample_counter = 0;
    }
    h->total_samples++;

    int idx = 0;
    if (h->bin_exp == 1.0f) {
        // uniform
        float t = (v - h->min_val) * h->bin_width_inv;
        idx = (int)t;
    } else {
        // exponential
        float norm = (v - h->min_val) * h->range_inv;
        if (norm <= 0.0f) {
            idx = 0;
        } else if (norm >= 1.0f) {
            idx = h->num_bins - 1;
        } else {
            // solve bin_exp^i = norm * denom + 1
            float arg = norm * h->denom + 1.0f;
            idx = (int)(logf(arg) / h->log_bin_exp);
        }
    }
    if (idx < 0) idx = 0;
    else if (idx >= h->num_bins) idx = h->num_bins - 1;

    // add weight
    h->bins[idx] += (1.0f - h->decay_factor);
}

float fast_histogram_quantile(fast_histogram_t *h, float p)
{
    assert(p >= 0.0f && p <= 1.0f);
    // compute total mass
    float total = 0.0f;
    for (int i = 0; i < h->num_bins; i++) total += h->bins[i];
    if (total <= 0.0f) return NAN;

    float target = p * total;
    float cum = 0.0f;
    for (int i = 0; i < h->num_bins; i++) {
        cum += h->bins[i];
        if (cum >= target) {
            // compute bin center
            if (h->bin_exp == 1.0f) {
                return h->min_val + (i + 0.5f) * h->bin_width;
            } else {
                // exponential bin center = midpoint of edges
                float total_range = h->max_val - h->min_val;
                float pi = expf(h->log_bin_exp * i);
                float pi1 = pi * h->bin_exp;
                float e0 = (pi  - 1.0f) / h->denom * total_range;
                float e1 = (pi1 - 1.0f) / h->denom * total_range;
                return h->min_val + 0.5f * (e0 + e1);
            }
        }
    }
    return h->max_val;
}

YAML::Node fast_histogram_get_stats(fast_histogram_t *h)
{
    YAML::Node root;
    root["samples"]    = h->total_samples;
    root["total"]= h->total;
    root["running_mean"]=h->running_mean;
    root["centile_50"] = fast_histogram_quantile(h, 0.5f);
    root["centile_90"] = fast_histogram_quantile(h, 0.9f);
    root["centile_99"] = fast_histogram_quantile(h, 0.99f);
    return root;
}