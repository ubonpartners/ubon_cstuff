// ------------------------------------------------------------
// Public Header Section (can split into audioio.h)
// ------------------------------------------------------------
#ifndef AUDIOIO_H_INCLUDED
#define AUDIOIO_H_INCLUDED

typedef enum {
    AUDIOIO_MODE_CAPTURE = 1,
    AUDIOIO_MODE_PLAYBACK = 2,
    AUDIOIO_MODE_FULL_DUPLEX = 3
} audioio_mode_t;

#include "audioframe.h"

// Forward decl
typedef struct audioio_stream audioio_stream_t;

// User callback types.
// Capture callback: invoked with newly available frame (ownership passed; user must destroy). Return 0 on success.
typedef int (*audioio_capture_cb)(audioframe_t *frame, void *userdata);
// Playback callback: asked to provide at least num_ms of audio. Return audioframe_t* or NULL if not ready.
typedef audioframe_t *(*audioio_playback_cb)(int num_ms, void *userdata);

// Stream parameter struct.
typedef struct audioio_stream_params {
    audioio_mode_t mode;
    const char *capture_device;   // e.g. "default" or "hw:0" (ignored if no capture)
    const char *playback_device;  // e.g. "default" (ignored if no playback)
    int sample_rate;              // desired sample rate (e.g. 16000 / 48000)
    int channels;                 // desired channel count (1 or 2 typical)
    int frame_ms;                 // granularity for produced/consumed frames (e.g. 10)
    int period_ms;                // ALSA hardware/software period size (suggest: frame_ms or a multiple). If 0 -> auto
    int buffer_ms;                // Overall ALSA buffer target (multiple of period). If 0 -> auto
    int use_float;                // Try to use ALSA FLOAT_LE if non-zero, else S16_LE
} audioio_stream_params_t;

// Create & destroy.
audioio_stream_t *audioio_stream_create(const audioio_stream_params_t *params);
void audioio_stream_destroy(audioio_stream_t *s);

// Optional callbacks (set before start or while running). Passing NULL disables.
int audioio_set_capture_callback(audioio_stream_t *s, audioio_capture_cb cb, void *ud);
int audioio_set_playback_callback(audioio_stream_t *s, audioio_playback_cb cb, void *ud);

// Start / Stop streaming (threads). Implicitly started on create unless start=0 used; here we auto-start.
int audioio_stream_start(audioio_stream_t *s);
int audioio_stream_stop(audioio_stream_t *s); // idempotent

// CAPTURE pull API: attempt to obtain at least num_ms of audio. Returns frame or NULL if not yet enough.
audioframe_t *audioio_capture_get_frame(audioio_stream_t *s, int num_ms);

// PLAYBACK push API: enqueue a frame for playback (takes ownership if refcount semantics are used; here we copy data).
int audioio_playback_write_frame(audioio_stream_t *s, audioframe_t *fr);

// Query for latency statistics (approximate).
int audioio_get_capture_lag_ms(audioio_stream_t *s);   // buffered but not yet consumed by user
int audioio_get_playback_queue_ms(audioio_stream_t *s); // queued frames pending playback

// Human readable error from last failing ALSA call stored in stream.
const char *audioio_strerror(audioio_stream_t *s);

#endif // AUDIOIO_H_INCLUDED