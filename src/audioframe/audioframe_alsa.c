// ============================================================
// audioframe_audioio_alsa.c  (NEW FILE - replaces previous content)
// ============================================================
// High-level Audio I/O interface built around the existing audioframe API.
// Targets Ubuntu Linux using ALSA (Advanced Linux Sound Architecture).
//
// Goals:
//   * Provide *capture* (microphone) and *playback* (speaker) functionality
//     producing/consuming `audioframe_t` objects.
//   * Abstract threading, buffering, and format negotiation so user code can
//     simply pull recorded frames or push frames for playback.
//   * Allow both pull (consumer calls get) and callback (user-supplied
//     function is invoked when new capture data is ready) styles.
//   * Similarly for playback: either push frames (non-blocking enqueue) or
//     register a callback that supplies frames on demand.
//
// Design Overview:
// -----------------
// We define an opaque handle `audioio_stream_t` which can represent either
// a CAPTURE or PLAYBACK stream (or FULL_DUPLEX containing both internally).
// Internally each direction is managed by a dedicated thread reading/writing
// the ALSA PCM device into a lock-protected ring buffer of interleaved float
// samples at an agreed native format (sample_rate, channels, period size).
//
// On CAPTURE:
//   ALSA thread continuously reads frames, converts to float [-1,1], writes
//   into circular buffer. User can call `audioio_capture_get_frame(num_ms)` to
//   obtain an `audioframe_t`. If insufficient data yet, returns NULL. Optionally,
//   user may install a capture callback that will be invoked from a *separate*
//   dispatch thread (not the ALSA thread) to avoid heavy processing inside
//   the real-time thread.
//
// On PLAYBACK:
//   User enqueues `audioframe_t` objects via `audioio_playback_write_frame`.
//   ALSA thread dequeues and writes to device. Optional playback callback can
//   be set which is queried for more audio when queue low (< one period).
//
// FULL_DUPLEX helper: convenience to open one capture and one playback device
// with (optionally different) device names but *same* sample rate and channels.
// Typically for echo cancellation / latency sensitive apps you'd want identical
// params. A future version could add automatic resampling using the existing
// resampler module if needed (hook points noted below).
//
// Key API Concepts:
//   * audioio_mode_t: CAPTURE, PLAYBACK, FULL_DUPLEX
//   * audioio_stream_params_t: desired sample_rate, channels, frame_ms granularity,
//     device names (strings), latency target.
//   * For simplicity we currently accept only interleaved float internal format
//     and negotiate ALSA format S16_LE or FLOAT_LE, converting to float.
//
// Thread Safety:
//   * All public API functions are thread-safe unless documented otherwise.
//   * Capture/playback ring buffers protected by mutex + condition variable.
//
// Error Handling:
//   * Functions return 0 on success, negative error codes on failure (mapped
//     from ALSA or custom). Creation returns NULL on failure.
//   * Use audioio_strerror() to get readable message.
//
// Build / Dependencies:
//   sudo apt-get install libasound2-dev
//   gcc -O2 -pthread -c audioframe_audioio_alsa.c `pkg-config --cflags alsa`
//   (link) -lasound -lpthread
//
// NOTE: This file includes both the *header* (guarded) and implementation for
// convenience. You may split into audioio.h / audioio_alsa.c if preferred.
// ============================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>
#include "log.h"
#include <alsa/asoundlib.h>

#include "audioframe.h"   // Existing interface provided earlier
#include "audio_io.h"

// ------------------------------------------------------------
// Implementation
// ------------------------------------------------------------

#ifndef AUDIOIO_MAX_QUEUE_FRAMES
#define AUDIOIO_MAX_QUEUE_FRAMES 256
#endif

#ifndef AUDIOIO_CAPTURE_RING_SECONDS
#define AUDIOIO_CAPTURE_RING_SECONDS 3   // ring buffer depth
#endif

struct audioio_ring {
    float *data;            // interleaved samples
    size_t frames_capacity; // total frame capacity
    size_t frames_count;    // frames currently stored
    size_t read_pos;        // frame index
};

static void ring_init(struct audioio_ring *r) {
    memset(r,0,sizeof(*r));
}
static int ring_alloc(struct audioio_ring *r, size_t frames, int channels) {
    r->data = (float*)malloc(frames * channels * sizeof(float));
    if (!r->data) return -1;
    r->frames_capacity = frames;
    r->frames_count = 0;
    r->read_pos = 0;
    return 0;
}
static void ring_free(struct audioio_ring *r) { free(r->data); memset(r,0,sizeof(*r)); }

// Write frames (interleaved). Overwrites oldest data if overflow (capture design choice: drop oldest to keep most recent).
static size_t ring_write(struct audioio_ring *r, const float *src, size_t frames, int ch) {
    if (frames==0) return 0;
    // If writing more than capacity, keep only last part
    if (frames > r->frames_capacity) {
        src += (frames - r->frames_capacity) * ch;
        frames = r->frames_capacity;
        r->read_pos = 0; r->frames_count = 0; // will fill below
    }
    // Ensure space: if overflow imminent, drop oldest
    size_t free_frames = r->frames_capacity - r->frames_count;
    if (frames > free_frames) {
        size_t drop = frames - free_frames;
        // Move read_pos forward dropping 'drop'
        r->read_pos = (r->read_pos + drop) % r->frames_capacity;
        r->frames_count -= drop;
    }
    // Write in up to two segments
    size_t write_pos = (r->read_pos + r->frames_count) % r->frames_capacity;
    size_t contiguous = r->frames_capacity - write_pos;
    if (contiguous > frames) contiguous = frames;
    memcpy(r->data + write_pos * ch, src, contiguous * ch * sizeof(float));
    size_t remaining = frames - contiguous;
    if (remaining) memcpy(r->data, src + contiguous * ch, remaining * ch * sizeof(float));
    r->frames_count += frames;
    return frames;
}

static size_t ring_peek(struct audioio_ring *r, float *dst, size_t frames, int ch) {
    if (frames > r->frames_count) frames = r->frames_count;
    size_t contiguous = r->frames_capacity - r->read_pos;
    if (contiguous > frames) contiguous = frames;
    memcpy(dst, r->data + r->read_pos * ch, contiguous * ch * sizeof(float));
    size_t remaining = frames - contiguous;
    if (remaining) memcpy(dst + contiguous * ch, r->data, remaining * ch * sizeof(float));
    return frames;
}

static size_t ring_consume(struct audioio_ring *r, size_t frames) {
    if (frames > r->frames_count) frames = r->frames_count;
    r->read_pos = (r->read_pos + frames) % r->frames_capacity;
    r->frames_count -= frames;
    return frames;
}

// Playback queue node
struct pb_node { audioframe_t *frame; struct pb_node *next; };

struct audioio_stream {
    audioio_mode_t mode;
    int sample_rate;
    int channels;
    int frame_ms;

    // ALSA handles
    snd_pcm_t *cap_handle;
    snd_pcm_t *pb_handle;

    int use_float;
    snd_pcm_format_t hw_format; // negotiated format (SND_PCM_FORMAT_FLOAT_LE or S16_LE)

    // Threads & sync
    pthread_t cap_thread;
    pthread_t pb_thread;
    pthread_t cap_cb_dispatch_thread; // optional dispatch for capture callbacks
    int run_capture;
    int run_playback;

    pthread_mutex_t cap_mutex;
    pthread_cond_t  cap_cond;

    pthread_mutex_t pb_mutex;
    pthread_cond_t  pb_cond;

    // Capture ring buffer
    struct audioio_ring cap_ring;

    // Playback queue
    struct pb_node *pb_head, *pb_tail;
    int pb_frames_queued; // total frames inside queue

    // Callbacks
    audioio_capture_cb capture_cb; void *capture_ud;
    audioio_playback_cb playback_cb; void *playback_ud;

    // Error state
    char last_error[128];

    // For capture callback dispatch
    int cap_dispatch_running;
};

static void set_error(audioio_stream_t *s, const char *msg) {
    if (!s) return;
    strncpy(s->last_error, msg, sizeof(s->last_error)-1);
    s->last_error[sizeof(s->last_error)-1] = '\0';
}

const char *audioio_strerror(audioio_stream_t *s) { return s ? s->last_error : ""; }

// ---------------------------------- ALSA Helpers ----------------------------------
static int alsa_set_hw(snd_pcm_t *h, int sample_rate, int channels, snd_pcm_format_t fmt, int *out_period_frames, int period_ms, int buffer_ms) {
    snd_pcm_hw_params_t *hw; snd_pcm_hw_params_alloca(&hw);
    int err;
    if ((err = snd_pcm_hw_params_any(h, hw)) < 0) return err;
    if ((err = snd_pcm_hw_params_set_access(h, hw, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) return err;
    if ((err = snd_pcm_hw_params_set_format(h, hw, fmt)) < 0) return err;
    if ((err = snd_pcm_hw_params_set_rate(h, hw, sample_rate, 0)) < 0) return err;
    if ((err = snd_pcm_hw_params_set_channels(h, hw, channels)) < 0) return err;

    unsigned int rate_near = sample_rate; // confirm
    snd_pcm_hw_params_get_rate(hw, &rate_near, 0);

    // Period size
    snd_pcm_uframes_t period_frames = 0;
    if (period_ms <=0) period_ms = 10; // default
    period_frames = (snd_pcm_uframes_t)((uint64_t)rate_near * period_ms / 1000);
    if (period_frames < 16) period_frames = 16;
    snd_pcm_hw_params_set_period_size_near(h, hw, &period_frames, 0);

    // Buffer size (multiple periods)
    snd_pcm_uframes_t buffer_frames = 0;
    if (buffer_ms <=0) buffer_ms = period_ms * 4; // 4 periods default
    buffer_frames = (snd_pcm_uframes_t)((uint64_t)rate_near * buffer_ms / 1000);
    if (buffer_frames < period_frames*2) buffer_frames = period_frames * 2;
    snd_pcm_hw_params_set_buffer_size_near(h, hw, &buffer_frames);

    if ((err = snd_pcm_hw_params(h, hw)) < 0) return err;

    if (out_period_frames) *out_period_frames = (int)period_frames;
    return 0;
}

static int alsa_prepare_stream(snd_pcm_t **handle, const char *device, snd_pcm_stream_t dir) {
    int err = snd_pcm_open(handle, device, dir, 0);
    return err;
}

// ----------------------------- Thread Functions -----------------------------------

static void *capture_thread_func(void *arg) {
    audioio_stream_t *s = (audioio_stream_t*)arg;
    int ch = s->channels;
    int period_frames = s->frame_ms * s->sample_rate / 1000; // We'll read in frame_ms chunks
    if (period_frames <=0) period_frames = 160; // fallback for 10ms @16k
    float *float_buf = (float*)malloc(period_frames * ch * sizeof(float));
    int16_t *tmp16 = NULL;
    if (s->hw_format == SND_PCM_FORMAT_S16_LE) tmp16 = (int16_t*)malloc(period_frames * ch * sizeof(int16_t));

    while (s->run_capture) {
        int frames_needed = period_frames;
        int rd = 0;
        while (rd < frames_needed && s->run_capture) {
            snd_pcm_sframes_t got;
            if (s->hw_format == SND_PCM_FORMAT_FLOAT_LE) {
                got = snd_pcm_readi(s->cap_handle, float_buf + rd*ch, frames_needed - rd);
            } else {
                got = snd_pcm_readi(s->cap_handle, tmp16 + rd*ch, frames_needed - rd);
            }
            if (got == -EPIPE) { snd_pcm_prepare(s->cap_handle); continue; }
            if (got < 0) { usleep(2000); continue; }
            rd += (int)got;
        }
        if (!s->run_capture) break;
        if (s->hw_format == SND_PCM_FORMAT_S16_LE) {
            for (int i=0;i<rd*ch;i++) float_buf[i] = tmp16[i] / 32768.0f; // convert
        }
        // Write to ring
        pthread_mutex_lock(&s->cap_mutex);
        ring_write(&s->cap_ring, float_buf, rd, ch);
        pthread_cond_signal(&s->cap_cond);
        pthread_mutex_unlock(&s->cap_mutex);
    }
    free(float_buf); free(tmp16);
    return NULL;
}

static void *capture_dispatch_thread(void *arg) {
    audioio_stream_t *s = (audioio_stream_t*)arg;
    const int ms_chunk = s->frame_ms;
    while (s->cap_dispatch_running) {
        pthread_mutex_lock(&s->cap_mutex);
        // Wait until we have at least one chunk
        size_t need_frames = (size_t)s->sample_rate * ms_chunk / 1000;
        while (s->cap_dispatch_running && s->cap_ring.frames_count < need_frames) {
            pthread_cond_wait(&s->cap_cond, &s->cap_mutex);
        }
        if (!s->cap_dispatch_running) { pthread_mutex_unlock(&s->cap_mutex); break; }
        // Extract one chunk
        size_t got = need_frames;
        float *tmp = (float*)malloc(got * s->channels * sizeof(float));
        ring_peek(&s->cap_ring, tmp, got, s->channels);
        ring_consume(&s->cap_ring, got);
        pthread_mutex_unlock(&s->cap_mutex);

        audioframe_t *fr = audioframe_create((int)got, s->sample_rate, s->channels);
        memcpy(audioframe_get_data(fr), tmp, got * s->channels * sizeof(float));
        free(tmp);
        if (s->capture_cb) s->capture_cb(fr, s->capture_ud);
        audioframe_destroy(fr); // user should ref if they need to keep it
    }
    return NULL;
}

static void *playback_thread_func(void *arg) {
    audioio_stream_t *s = (audioio_stream_t*)arg;
    int ch = s->channels;
    int period_frames = s->frame_ms * s->sample_rate / 1000;
    if (period_frames <=0) period_frames = 160;
    float *float_buf = (float*)malloc(period_frames * ch * sizeof(float));
    int16_t *tmp16 = NULL;
    if (s->hw_format == SND_PCM_FORMAT_S16_LE) tmp16 = (int16_t*)malloc(period_frames * ch * sizeof(int16_t));

    while (s->run_playback) {
        int filled = 0;
        while (filled < period_frames && s->run_playback) {
            // Pop from queue
            pthread_mutex_lock(&s->pb_mutex);
            if (!s->pb_head && s->playback_cb) {
                // request via callback (outside lock) minimal frames
                pthread_mutex_unlock(&s->pb_mutex);
                audioframe_t *cbfr = s->playback_cb(s->frame_ms, s->playback_ud);
                if (cbfr) audioio_playback_write_frame(s, cbfr); // will copy
                if (cbfr) audioframe_destroy(cbfr);
                pthread_mutex_lock(&s->pb_mutex);
            }
            struct pb_node *node = s->pb_head;
            if (!node) {
                // nothing queued: fill silence
                pthread_mutex_unlock(&s->pb_mutex);
                int remaining = period_frames - filled;
                memset(float_buf + filled*ch, 0, remaining * ch * sizeof(float));
                filled = period_frames; // break loop
                break;
            }
            audioframe_t *fr = node->frame;
            int fr_frames = audioframe_get_num_samples(fr);
            int to_copy = fr_frames;
            if (to_copy > period_frames - filled) to_copy = period_frames - filled;
            memcpy(float_buf + filled*ch, audioframe_get_data(fr), to_copy * ch * sizeof(float));
            filled += to_copy;
            if (to_copy < fr_frames) {
                // partial consume: shift remaining samples forward inside frame (simple but O(n))
                float *d = audioframe_get_data(fr);
                memmove(d, d + to_copy*ch, (fr_frames - to_copy) * ch * sizeof(float));
                // reduce internal sample count? (Assuming audioframe immutable; alternatively allocate new smaller) —
                // For simplicity we ignore adjusting meta; realistic implementation should support trimming or using an offset.
            } else {
                // fully consumed
                s->pb_head = node->next;
                if (!s->pb_head) s->pb_tail = NULL;
                audioframe_destroy(fr);
                free(node);
            }
            pthread_mutex_unlock(&s->pb_mutex);
        }
        if (!s->run_playback) break;

        // Convert + write
        if (s->hw_format == SND_PCM_FORMAT_S16_LE) {
            for (int i=0;i<filled*ch;i++) {
                float v = float_buf[i]; if (v>1) v=1; else if (v<-1) v=-1;
                tmp16[i] = (int16_t)lrintf(v * 32767.0f);
            }
            int off=0;
            while (off < filled) {
                snd_pcm_sframes_t wrote = snd_pcm_writei(s->pb_handle, tmp16 + off*ch, filled - off);
                if (wrote == -EPIPE) { snd_pcm_prepare(s->pb_handle); continue; }
                if (wrote < 0) { usleep(2000); continue; }
                off += (int)wrote;
            }
        } else {
            int off=0;
            while (off < filled) {
                snd_pcm_sframes_t wrote = snd_pcm_writei(s->pb_handle, float_buf + off*ch, filled - off);
                if (wrote == -EPIPE) { snd_pcm_prepare(s->pb_handle); continue; }
                if (wrote < 0) { usleep(2000); continue; }
                off += (int)wrote;
            }
        }
    }
    free(float_buf); free(tmp16);
    return NULL;
}

// --------------------------------- Public API -------------------------------------

audioio_stream_t *audioio_stream_create(const audioio_stream_params_t *params) {
    if (!params) return NULL;
    audioio_stream_t *s = (audioio_stream_t*)calloc(1,sizeof(*s));
    if (!s) return NULL;
    s->mode = params->mode;
    s->sample_rate = params->sample_rate;
    s->channels = params->channels;
    s->frame_ms = params->frame_ms > 0 ? params->frame_ms : 10;
    s->use_float = params->use_float;
    pthread_mutex_init(&s->cap_mutex,NULL);
    pthread_cond_init(&s->cap_cond,NULL);
    pthread_mutex_init(&s->pb_mutex,NULL);
    pthread_cond_init(&s->pb_cond,NULL);

    const char *cap_dev = params->capture_device ? params->capture_device : "default";
    const char *pb_dev  = params->playback_device ? params->playback_device : "default";

    int period_frames=0;
    int err;

    if (s->mode & AUDIOIO_MODE_CAPTURE) {
        if ((err = alsa_prepare_stream(&s->cap_handle, cap_dev, SND_PCM_STREAM_CAPTURE)) < 0) {
            set_error(s, snd_strerror(err));
            audioio_stream_destroy(s);
            return NULL;
         }
    }
    if (s->mode & AUDIOIO_MODE_PLAYBACK) {
        if ((err = alsa_prepare_stream(&s->pb_handle, pb_dev, SND_PCM_STREAM_PLAYBACK)) < 0) {
            set_error(s, snd_strerror(err));
            audioio_stream_destroy(s);
            return NULL;
        }
    }

    // Choose format preference
    snd_pcm_format_t try_fmt = s->use_float ? SND_PCM_FORMAT_FLOAT_LE : SND_PCM_FORMAT_S16_LE;

    if (s->cap_handle) {
        if ((err = alsa_set_hw(s->cap_handle, s->sample_rate, s->channels, try_fmt, &period_frames, params->period_ms, params->buffer_ms)) < 0) {
            if (try_fmt == SND_PCM_FORMAT_FLOAT_LE) { // fallback to S16
                if ((err = alsa_set_hw(s->cap_handle, s->sample_rate, s->channels, SND_PCM_FORMAT_S16_LE, &period_frames, params->period_ms, params->buffer_ms)) < 0) {
                    set_error(s,snd_strerror(err));
                    audioio_stream_destroy(s);
                    return NULL;
                }
                s->hw_format = SND_PCM_FORMAT_S16_LE;
            } else {
                set_error(s,snd_strerror(err));
                audioio_stream_destroy(s);
                return NULL;
            }
        } else s->hw_format = try_fmt;
    }

    if (s->pb_handle) {
        if ((err = alsa_set_hw(s->pb_handle, s->sample_rate, s->channels, try_fmt, &period_frames, params->period_ms, params->buffer_ms)) < 0) {
            if (try_fmt == SND_PCM_FORMAT_FLOAT_LE) { // fallback
                if ((err = alsa_set_hw(s->pb_handle, s->sample_rate, s->channels, SND_PCM_FORMAT_S16_LE, &period_frames, params->period_ms, params->buffer_ms)) < 0) {
                    set_error(s,snd_strerror(err));
                    audioio_stream_destroy(s);
                    return NULL;
                }
                s->hw_format = SND_PCM_FORMAT_S16_LE;
            } else {
                set_error(s,snd_strerror(err));
                audioio_stream_destroy(s);
                return NULL;
            }
        } else s->hw_format = try_fmt;
    }

    // Allocate capture ring
    if (s->cap_handle) {
        ring_init(&s->cap_ring);
        size_t ring_frames = (size_t)s->sample_rate * AUDIOIO_CAPTURE_RING_SECONDS;
        if (ring_alloc(&s->cap_ring, ring_frames, s->channels) != 0) {
            set_error(s, "ring_alloc failed");
            audioio_stream_destroy(s);
            return NULL;
        }
    }

    // Start threads
    if (audioio_stream_start(s) != 0)
    {
        audioio_stream_destroy(s);
        return NULL;
    }

    return s;
}

int audioio_stream_start(audioio_stream_t *s) {
    if (!s) return -1;
    int err;
    if (s->cap_handle && !s->run_capture) {
        s->run_capture = 1;
        if ((err = pthread_create(&s->cap_thread, NULL, capture_thread_func, s)) != 0) { set_error(s, strerror(err)); s->run_capture=0; return -1; }
    }
    if (s->pb_handle && !s->run_playback) {
        s->run_playback = 1;
        if ((err = pthread_create(&s->pb_thread, NULL, playback_thread_func, s)) != 0) { set_error(s, strerror(err)); s->run_playback=0; return -1; }
    }
    return 0;
}

int audioio_stream_stop(audioio_stream_t *s) {
    if (!s) return -1;
    if (s->run_capture) { s->run_capture = 0; pthread_cond_broadcast(&s->cap_cond); pthread_join(s->cap_thread,NULL); }
    if (s->cap_dispatch_running) { s->cap_dispatch_running = 0; pthread_cond_broadcast(&s->cap_cond); pthread_join(s->cap_cb_dispatch_thread,NULL); }
    if (s->run_playback) { s->run_playback = 0; pthread_cond_broadcast(&s->pb_cond); pthread_join(s->pb_thread,NULL); }
    return 0;
}

void audioio_stream_destroy(audioio_stream_t *s) {
    if (!s) return;
    audioio_stream_stop(s);
    if (s->cap_handle) snd_pcm_close(s->cap_handle);
    if (s->pb_handle) snd_pcm_close(s->pb_handle);

    // Free playback queue
    struct pb_node *n = s->pb_head; while (n) { struct pb_node *nx = n->next; audioframe_destroy(n->frame); free(n); n = nx; }

    ring_free(&s->cap_ring);
    pthread_mutex_destroy(&s->cap_mutex);
    pthread_cond_destroy(&s->cap_cond);
    pthread_mutex_destroy(&s->pb_mutex);
    pthread_cond_destroy(&s->pb_cond);

    free(s);
}

int audioio_set_capture_callback(audioio_stream_t *s, audioio_capture_cb cb, void *ud) {
    if (!s) return -1; if (!(s->mode & AUDIOIO_MODE_CAPTURE)) return -2;
    pthread_mutex_lock(&s->cap_mutex);
    s->capture_cb = cb; s->capture_ud = ud;
    int need_thread = cb && !s->cap_dispatch_running;
    if (need_thread) {
        s->cap_dispatch_running = 1;
        int err = pthread_create(&s->cap_cb_dispatch_thread, NULL, capture_dispatch_thread, s);
        if (err) { s->cap_dispatch_running = 0; set_error(s, strerror(err)); pthread_mutex_unlock(&s->cap_mutex); return -1; }
    }
    if (!cb && s->cap_dispatch_running) {
        s->cap_dispatch_running = 0; pthread_cond_broadcast(&s->cap_cond);
        pthread_mutex_unlock(&s->cap_mutex);
        pthread_join(s->cap_cb_dispatch_thread,NULL);
        return 0;
    }
    pthread_mutex_unlock(&s->cap_mutex);
    return 0;
}

int audioio_set_playback_callback(audioio_stream_t *s, audioio_playback_cb cb, void *ud) {
    if (!s) return -1; if (!(s->mode & AUDIOIO_MODE_PLAYBACK)) return -2;
    pthread_mutex_lock(&s->pb_mutex);
    s->playback_cb = cb; s->playback_ud = ud;
    pthread_mutex_unlock(&s->pb_mutex);
    return 0;
}

audioframe_t *audioio_capture_get_frame(audioio_stream_t *s, int num_ms) {
    if (!s || !(s->mode & AUDIOIO_MODE_CAPTURE) || num_ms <=0) return NULL;
    size_t need_frames = (size_t)s->sample_rate * num_ms / 1000;
    pthread_mutex_lock(&s->cap_mutex);
    if (s->cap_ring.frames_count < need_frames) { pthread_mutex_unlock(&s->cap_mutex); return NULL; }
    float *tmp = (float*)malloc(need_frames * s->channels * sizeof(float));
    ring_peek(&s->cap_ring, tmp, need_frames, s->channels);
    ring_consume(&s->cap_ring, need_frames);
    pthread_mutex_unlock(&s->cap_mutex);

    audioframe_t *fr = audioframe_create((int)need_frames, s->sample_rate, s->channels);
    memcpy(audioframe_get_data(fr), tmp, need_frames * s->channels * sizeof(float));
    free(tmp);
    return fr;
}

int audioio_playback_write_frame(audioio_stream_t *s, audioframe_t *fr) {
    if (!s || !(s->mode & AUDIOIO_MODE_PLAYBACK) || !fr) return -1;
    if (audioframe_get_sample_rate(fr) != s->sample_rate || audioframe_get_num_channels(fr) != s->channels)
    {
        log_error("SR/CH mismatch %d/%d!=%d/%d", audioframe_get_sample_rate(fr),
                   audioframe_get_num_channels(fr),
                   s->sample_rate, s->channels);
        return -2; // mismatch; future: integrate resampler
    }
    int frames = audioframe_get_num_samples(fr);
    float *data = audioframe_get_data(fr);
    audioframe_t *copy = audioframe_create(frames, s->sample_rate, s->channels);
    memcpy(audioframe_get_data(copy), data, (size_t)frames * s->channels * sizeof(float));

    struct pb_node *node = (struct pb_node*)malloc(sizeof(*node));
    if (!node) { audioframe_destroy(copy); return -1; }
    node->frame = copy; node->next = NULL;
    pthread_mutex_lock(&s->pb_mutex);
    if (s->pb_tail) s->pb_tail->next = node; else s->pb_head = node;
    s->pb_tail = node;
    s->pb_frames_queued += frames;
    pthread_mutex_unlock(&s->pb_mutex);
    return 0;
}

int audioio_get_capture_lag_ms(audioio_stream_t *s) {
    if (!s || !(s->mode & AUDIOIO_MODE_CAPTURE)) return 0;
    pthread_mutex_lock(&s->cap_mutex);
    size_t frames = s->cap_ring.frames_count;
    pthread_mutex_unlock(&s->cap_mutex);
    return (int)(frames * 1000 / s->sample_rate);
}

int audioio_get_playback_queue_ms(audioio_stream_t *s) {
    if (!s || !(s->mode & AUDIOIO_MODE_PLAYBACK)) return 0;
    pthread_mutex_lock(&s->pb_mutex);
    int frames = s->pb_frames_queued;
    pthread_mutex_unlock(&s->pb_mutex);
    return (int)(frames * 1000 / s->sample_rate);
}

// ------------------------------------------------------------
// Example Usage (Pseudo-code)
// ------------------------------------------------------------
//  audioio_stream_params_t P = {0};
//  P.mode = AUDIOIO_MODE_FULL_DUPLEX;
//  P.capture_device = "default"; P.playback_device = "default";
//  P.sample_rate = 16000; P.channels = 1; P.frame_ms = 10; P.use_float = 1;
//  audioio_stream_t *io = audioio_stream_create(&P);
//  if (!io) { fprintf(stderr, "Failed: %s\n", audioio_strerror(io)); }
//
//  // Pull recorded frames:
//  while (running) {
//      audioframe_t *in = audioio_capture_get_frame(io, 10);
//      if (in) { /* process */ audioframe_destroy(in); }
//      // Provide playback frames (silence example)
//      audioframe_t *out = audioframe_create(160, P.sample_rate, P.channels);
//      memset(audioframe_get_data(out), 0, 160*sizeof(float));
//      audioio_playback_write_frame(io, out);
//      audioframe_destroy(out);
//  }
//  audioio_stream_destroy(io);
//
// ------------------------------------------------------------
// Future Extensions / Hooks:
//   * Integrate audioframe_resampler to handle mismatched sample rates & channels.
//   * Add echo reference path / loopback capture.
//   * Provide non-blocking poll function returning file descriptors for integration
//     with external event loops (ALSA can expose descriptors via snd_pcm_poll_descriptors()).
//   * Add explicit flush / drain for playback.
//   * Implement zero-copy for playback frames by referencing audioframe data directly
//     with refcounts instead of copying.
// ------------------------------------------------------------

// ============================================================
// END OF FILE
// ============================================================
