// ------------------------------------------------------------
// audioframe_resampler.c  (libsamplerate-based high-quality version)
// ------------------------------------------------------------
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <samplerate.h>
#include "audioframe.h"

#ifndef RESAMPLER_MAX_EXTRA_FACTOR
#define RESAMPLER_MAX_EXTRA_FACTOR 2   // limit how many extra ms we produce per call
#endif

struct audioframe_resampler {
    int target_sample_rate;
    int target_channels;

    SRC_STATE *state;
    int src_sample_rate_current;

    float *input_buffer;    // interleaved, target_channels
    int    input_frames;
    int    input_capacity;

    float *output_accum;    // accumulate until we reach threshold
    int    output_frames;
    int    output_capacity;
};

static void *xrealloc(void *p, size_t sz) {
    void *n = realloc(p, sz);
    if (!n) {
        free(p);
        return NULL;
    }
    return n;
}

static int ensure_capacity(float **buf, int *cap, int needed_frames, int channels) {
    if (needed_frames <= *cap) return 0;
    int new_cap = *cap ? *cap : 1024;
    while (new_cap < needed_frames) new_cap *= 2;
    float *nb = (float*)realloc(*buf, (size_t)new_cap * channels * sizeof(float));
    if (!nb) return -1;
    *buf = nb;
    *cap = new_cap;
    return 0;
}

audioframe_resampler_t *audioframe_resampler_create(int target_sample_rate, int target_num_channels) {
    if (target_sample_rate <=0 || target_num_channels <=0) return NULL;
    audioframe_resampler_t *r = (audioframe_resampler_t*)calloc(1,sizeof(*r));
    if (!r) return NULL;
    int err = 0;
    r->state = src_new(SRC_SINC_FASTEST, target_num_channels, &err);
    if (!r->state || err) {
        free(r);
        return NULL;
    }
    r->target_sample_rate = target_sample_rate;
    r->target_channels = target_num_channels;
    return r;
}

void audioframe_resampler_destroy(audioframe_resampler_t *r) {
    if (!r) return;
    if (r->state) src_delete(r->state);
    free(r->input_buffer);
    free(r->output_accum);
    free(r);
}

static void channel_convert(const float *in, int in_ch, float *out, int out_ch, int frames) {
    if (in_ch == out_ch) {
        memcpy(out, in, (size_t)frames * in_ch * sizeof(float));
        return;
    }
    for (int f=0; f<frames; ++f) {
        if (in_ch == 1 && out_ch > 1) {
            float v = in[f];
            for (int c=0;c<out_ch;c++) out[f*out_ch + c] = v;
        } else if (in_ch > 1 && out_ch == 1) {
            float sum = 0.f;
            for (int c=0;c<in_ch;c++) sum += in[f*in_ch + c];
            out[f] = sum / (float)in_ch;
        } else if (in_ch > 1 && out_ch > 1) {
            // down or up with proportional grouping
            for (int oc=0; oc<out_ch; ++oc) {
                int start = (oc * in_ch) / out_ch;
                int end   = ((oc+1) * in_ch) / out_ch;
                if (end <= start) end = start + 1;
                float sum = 0.f;
                for (int ic=start; ic<end; ++ic) sum += in[f*in_ch + ic];
                out[f*out_ch + oc] = sum / (float)(end - start);
            }
        } else {
            // Should not reach
            for (int c=0;c<out_ch;c++) out[f*out_ch+c] = 0.f;
        }
    }
}

audioframe_t *audioframe_resample(audioframe_resampler_t *r, audioframe_t *in_fr, int num_ms) {
    if (!r || num_ms <= 0) return NULL;

    // Append new input (after channel conversion) to input_buffer
    if (in_fr) {
        int in_rate = audioframe_get_sample_rate(in_fr);
        int in_ch   = audioframe_get_num_channels(in_fr);
        int in_frames = audioframe_get_num_samples(in_fr);
        const float *in_data = audioframe_get_data(in_fr);

        if (r->src_sample_rate_current != 0 && in_rate != r->src_sample_rate_current) {
            // Reset the SRC state for new sample rate
            src_reset(r->state);
            r->input_frames = 0;
            r->output_frames = 0;
        }
        r->src_sample_rate_current = in_rate;

        if (ensure_capacity(&r->input_buffer, &r->input_capacity, r->input_frames + in_frames, r->target_channels) != 0)
            return NULL;

        channel_convert(in_data, in_ch, r->input_buffer + (size_t)r->input_frames * r->target_channels,
                        r->target_channels, in_frames);
        r->input_frames += in_frames;
    }

    if (r->src_sample_rate_current == 0) return NULL; // nothing yet

    int needed_out_frames = (int)ceil((double)r->target_sample_rate * (double)num_ms / 1000.0);
    if (needed_out_frames <= 0) needed_out_frames = 1;
    int max_out_frames = needed_out_frames * RESAMPLER_MAX_EXTRA_FACTOR;

    // Process while we can and haven't reached needed_out_frames
    while (r->output_frames < needed_out_frames && r->input_frames > 0) {
        // Provide some or all of input to src_process
        SRC_DATA d;
        memset(&d, 0, sizeof(d));
        d.data_in = r->input_buffer;
        d.input_frames = r->input_frames;
        // Provide output space (temporary local buffer) — choose chunk size
        int request = (needed_out_frames - r->output_frames);
        if (request < 256) request = 256; // some minimum
        if (request > max_out_frames - r->output_frames)
            request = max_out_frames - r->output_frames;
        if (request <= 0) break;

        if (ensure_capacity(&r->output_accum, &r->output_capacity,
                            r->output_frames + request, r->target_channels) != 0)
            return NULL;

        d.data_out = r->output_accum + (size_t)r->output_frames * r->target_channels;
        d.output_frames = request;
        d.src_ratio = (double)r->target_sample_rate / (double)r->src_sample_rate_current;
        d.end_of_input = 0;

        int err = src_process(r->state, &d);
        if (err) {
            // On error, abort
            return NULL;
        }

        // Advance consumed input
        if (d.input_frames_used > 0) {
            int remain = r->input_frames - d.input_frames_used;
            if (remain > 0) {
                memmove(r->input_buffer,
                        r->input_buffer + (size_t)d.input_frames_used * r->target_channels,
                        (size_t)remain * r->target_channels * sizeof(float));
            }
            r->input_frames = remain;
        }

        r->output_frames += d.output_frames_gen;

        // If no progress (no frames used and produced), break to avoid infinite loop
        if (d.output_frames_gen == 0 && d.input_frames_used == 0)
            break;
    }

    if (r->output_frames < needed_out_frames)
        return NULL; // not enough yet

    // Produce audioframe_t with available output (we can choose to emit exactly needed or all)
    int emit_frames = r->output_frames;
    audioframe_t *out = audioframe_create(emit_frames, r->target_sample_rate, r->target_channels);
    if (!out) return NULL;
    memcpy(audioframe_get_data(out), r->output_accum,
           (size_t)emit_frames * r->target_channels * sizeof(float));

    // Reset accumulation (we emitted all)
    r->output_frames = 0;

    return out;
}
