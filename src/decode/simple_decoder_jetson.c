#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <sys/prctl.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#if (UBONCSTUFF_PLATFORM == 1) // Orin Nano

#include "cudaEGL.h"
#include "NvUtils.h"
#include "NvVideoDecoder.h"
#include "NvBufSurface.h"
#include "simple_decoder.h"
#include "cuda_stuff.h"
#include "yaml_stuff.h"
#include "profile.h"

#define debugf if (0) log_error
#define CHECK_ERROR(cond, msg) do { if (cond) { log_fatal("%s", msg); abort(); } } while (0)

#define NR_CAPTURE_BUFFERS   10
#define MAX_RESCHANGES       30
#define CHUNK_SIZE           (4 * 1024 * 1024)
#define NR_OUTPUT_BUFFERS    2

// Context and config structures

typedef struct {
    int fd, index;
    simple_decoder_t *decoder;
    struct timeval timestamp;
} dma_fd_t;

typedef struct {
    uint32_t dec_w, dec_h;
    uint32_t win_w, win_h, win_x, win_y;
    uint32_t pix_fmt;
    int out_buf_count;
    int cap_buf_count;
    float fps;
    int fullscreen, disable_render;
} decctx_cfg_t;

// Decoder internals

struct simple_decoder {
    NvVideoDecoder *nvdec;
    decctx_cfg_t cfg;
    uint32_t pixfmt;
    int min_cap_buffers;

    pthread_t thread_id;
    bool got_error, got_eos, low_latency;

    int width, height;           // decoded dims
    int out_width, out_height;   // display dims

    image_format_t out_fmt;
    void *ctx;
    void (*frame_cb)(void*, image_t*);

    int max_w, max_h;
    double min_delta, last_time;
    int scaled_w, scaled_h;

    uint64_t total_bytes;
    uint32_t decode_calls;
    uint32_t outputs, skips, resets;
};

extern int nvSurfToImageNV12Device(NvBufSurface *nvSurf,image_t *img, CUstream stream);

static void abort_decoder(simple_decoder_t *d) {
    log_fatal("Decoder abort");
    d->got_error = true;
    d->nvdec->abort();
    abort();
}

static void apply_capture_settings(simple_decoder_t *d) {
    NvVideoDecoder *dec = d->nvdec;
    struct v4l2_format fmt;
    struct v4l2_crop crop;

    CHECK_ERROR(dec->capture_plane.getFormat(fmt) < 0, "Cannot get capture format");
    CHECK_ERROR(dec->capture_plane.getCrop(crop)   < 0, "Cannot get capture crop");

    d->cfg.dec_w = crop.c.width;
    d->cfg.dec_h = crop.c.height;
    d->cfg.win_w = crop.c.width;
    d->cfg.win_h = crop.c.height;
    d->cfg.pix_fmt = V4L2_PIX_FMT_YUV420M;
    d->pixfmt     = V4L2_PIX_FMT_YUV420M;
    log_info("Resolution: %ux%u", d->cfg.dec_w, d->cfg.dec_h);

    dec->capture_plane.deinitPlane();
    CHECK_ERROR(dec->setCapturePlaneFormat(fmt.fmt.pix_mp.pixelformat,
                                          fmt.fmt.pix_mp.width,
                                          fmt.fmt.pix_mp.height) < 0,
                "Failed to set capture format");

    int min_bufs;
    CHECK_ERROR(dec->getMinimumCapturePlaneBuffers(min_bufs) < 0, "Failed to get min buffers");
    CHECK_ERROR(min_bufs > NR_CAPTURE_BUFFERS, "Min buffers exceed limit");
    d->min_cap_buffers = min_bufs;

    int req = d->low_latency ? min_bufs : (min_bufs + 1);
    CHECK_ERROR(dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP, req, false, false) < 0, "Capture plane setup failed");

    d->cfg.cap_buf_count = dec->capture_plane.getNumBuffers();
    CHECK_ERROR(dec->capture_plane.setStreamStatus(true) < 0, "Failed to start capture stream");

    for (uint32_t i = 0; i < (uint32_t)d->cfg.cap_buf_count; ++i) {
        struct v4l2_buffer buf = {};
        struct v4l2_plane planes[MAX_PLANES] = {};
        buf.index = i;
        buf.m.planes = planes;
        CHECK_ERROR(dec->capture_plane.qBuffer(buf, NULL) < 0, "Queue capture buffer failed");
    }
}

static int wait_resolution_change(simple_decoder_t *d) {
    NvVideoDecoder *dec = d->nvdec;
    struct v4l2_event ev;
    for (int i = 0; i < MAX_RESCHANGES; ++i) {
        int ret = dec->dqEvent(ev, 1000);
        if (ret < 0) {
            if (d->got_eos) return 0;
            if (errno == EAGAIN) continue;
            return ret;
        }
        if (ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
            //log_info("Resolution change event");
            return 0;
        }
    }
    return -1;
}

static void handle_frame(simple_decoder_t *d, NvBuffer *buf, double t, bool force_skip) {
    if (d->min_delta > 0) {
        double delta = t - d->last_time;
        if (delta < 0 || delta > 10.0) {
            d->last_time = t;
            d->resets++;
        } else if (delta < d->min_delta) {
            d->skips++;
            return;
        }
    }

    NvBufSurface *surf = nullptr;
    if (NvBufSurfaceFromFd(buf->planes[0].fd, (void**)&surf) != 0) {
        log_error("Failed NvBufSurfaceFromFd");
        return;
    }

    image_t *img = image_create(d->cfg.dec_w, d->cfg.dec_h, IMAGE_FORMAT_NV12_DEVICE);
    if (nvSurfToImageNV12Device(surf, img, img->stream) != 0) {
        log_error("nvSurfToImageNV12Device error");
    }

    img->meta.time = t;
    img->meta.capture_realtime = profile_time();
    img->meta.flags = MD_CAPTURE_REALTIME_SET;

    d->last_time = t;

    image_sync(img);
    image_t *out = img;
    if (d->max_w && d->max_h) {
        determine_scale_size(img->width, img->height,
                             d->max_w, d->max_h,
                             &d->scaled_w, &d->scaled_h,
                             10, 8, 8, false);
        out = image_scale(img, d->scaled_w, d->scaled_h);
    }

    d->outputs++;
    d->frame_cb(d->ctx, out);
    image_destroy(out);
    image_destroy(img);
}

static void *capture_thread(void *arg) {
    cuda_thread_init();
    simple_decoder_t *d = (simple_decoder_t*)arg;
    if (wait_resolution_change(d) < 0) abort_decoder(d);
    apply_capture_settings(d);

    while (!d->got_error && !d->got_eos && !d->nvdec->isInError()) {
        struct v4l2_event ev;
        if (d->nvdec->dqEvent(ev, false) == 0 && ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
            apply_capture_settings(d);
            continue;
        }

        struct v4l2_buffer buf = {};
        struct v4l2_plane planes[MAX_PLANES] = {};
        buf.m.planes = planes;
        NvBuffer *nbuf;

        int ret = d->nvdec->capture_plane.dqBuffer(buf, &nbuf, NULL, 0);
        if (ret) {
            if (d->got_eos || errno == EPIPE) break;
            if (errno == EAGAIN) { usleep(1000); continue; }
            abort_decoder(d);
        }

        bool skip = (buf.timestamp.tv_usec & 3) != 0;
        double t = buf.timestamp.tv_sec + (buf.timestamp.tv_usec & ~3) * 1e-6;
        handle_frame(d, nbuf, t, skip);

        if (!d->got_eos) {
            CHECK_ERROR(d->nvdec->capture_plane.qBuffer(buf, NULL) < 0, "Requeue capture buffer failed");
        }
    }
    return NULL;
}

simple_decoder_t *simple_decoder_create(void *ctx,
                                       void (*cb)(void*, image_t*),
                                       simple_decoder_codec_t codec,
                                       bool low_latency) {
    simple_decoder_t *d = (simple_decoder_t*)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->ctx = ctx;
    d->frame_cb = cb;
    d->low_latency = low_latency;
    d->pixfmt = (codec==SIMPLE_DECODER_CODEC_H265) ? V4L2_PIX_FMT_H265 : V4L2_PIX_FMT_H264;
    d->nvdec = NvVideoDecoder::createVideoDecoder("dec0");
    CHECK_ERROR(!d->nvdec, "Failed to create decoder");
    CHECK_ERROR(d->nvdec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE,0,0)<0, "Event subscription failed");
    CHECK_ERROR(d->nvdec->setOutputPlaneFormat(d->pixfmt, CHUNK_SIZE)<0, "Set output format failed");
    CHECK_ERROR(d->nvdec->setFrameInputMode(1)<0, "Set frame mode failed");
    CHECK_ERROR(d->nvdec->disableDPB()<0, "Disable DPB failed");
    CHECK_ERROR(d->nvdec->setSkipFrames(V4L2_SKIP_FRAMES_TYPE_NONREF)<0, "Skip frames setup failed");
    CHECK_ERROR(d->nvdec->output_plane.setupPlane(V4L2_MEMORY_MMAP, NR_OUTPUT_BUFFERS, true, false)<0, "Output plane setup failed");
    CHECK_ERROR(d->nvdec->setMaxPerfMode(1)<0, "Max perf mode failed");
    CHECK_ERROR(d->nvdec->output_plane.setStreamStatus(true)<0, "Output streamon failed");

    d->cfg.out_buf_count = d->nvdec->output_plane.getNumBuffers();
    log_info("Output buffers: %d", d->cfg.out_buf_count);

    pthread_create(&d->thread_id, NULL, capture_thread, d);
    return d;
}

void simple_decoder_destroy(simple_decoder_t *d) {
    if (!d) return;
    d->got_eos = true;
    d->nvdec->abort();
    pthread_join(d->thread_id, NULL);

    d->nvdec->capture_plane.setStreamStatus(false);
    d->nvdec->capture_plane.deinitPlane();
    d->nvdec->output_plane.setStreamStatus(false);

    struct v4l2_buffer buf = {};
    struct v4l2_plane planes[MAX_PLANES] = {};
    while (d->nvdec->output_plane.getNumQueuedBuffers() > 0) {
        buf.m.planes = planes;
        if (d->nvdec->output_plane.dqBuffer(buf, NULL, NULL, 0) < 0) break;
    }

    delete d->nvdec;
    free(d);
    log_info("Decoder cleaned up");
}

void simple_decoder_decode(simple_decoder_t *d,
                           uint8_t *data, int size,
                           double ts, bool skip) {
    d->decode_calls++;
    d->total_bytes += size;

    struct v4l2_buffer buf = {};
    struct v4l2_plane planes[MAX_PLANES] = {};
    buf.m.planes = planes;

    NvBuffer *nbuf;
    if (d->decode_calls <= d->cfg.out_buf_count) {
        buf.index = d->decode_calls - 1;
        nbuf = d->nvdec->output_plane.getNthBuffer(buf.index);
    } else {
        CHECK_ERROR(d->nvdec->output_plane.dqBuffer(buf, &nbuf, NULL, -1) < 0,
                    "Dequeue output buffer failed");
    }

    memcpy(nbuf->planes[0].data, data, size);
    nbuf->planes[0].bytesused = size;
    buf.m.planes[0].bytesused = size;
    buf.flags = V4L2_BUF_FLAG_TIMESTAMP_COPY;
    buf.timestamp.tv_sec  = (time_t)ts;
    buf.timestamp.tv_usec = ((long)((ts - buf.timestamp.tv_sec)*1e6) & ~3) | (skip?3:0);

    CHECK_ERROR(d->nvdec->output_plane.qBuffer(buf, NULL) < 0,
                "Queue output buffer failed");
}

void simple_decoder_set_output_format(simple_decoder_t *d, image_format_t fmt) {
    if (d) d->out_fmt = fmt;
}

void simple_decoder_constrain_output(simple_decoder_t *d,
                                    int max_w, int max_h,
                                    double min_dt) {
    d->max_w = max_w;
    d->max_h = max_h;
    d->min_delta = min_dt;
}

YAML::Node simple_decoder_get_stats(simple_decoder_t *d) {
    YAML::Node n;
    n["bytes_decoded"]       = d->total_bytes;
    n["decode_calls"]        = d->decode_calls;
    n["surfaces_output"]     = d->outputs;
    n["frames_skipped"]      = d->skips;
    n["time_resets"]         = d->resets;
    return n;
}

#endif // UBONCSTUFF_PLATFORM
