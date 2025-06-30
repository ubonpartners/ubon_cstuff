
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include "simple_decoder.h"
#include "cuda_stuff.h"

#if (UBONCSTUFF_PLATFORM == 1) // Orin Nano
struct simple_decoder
{
    int width;
    int height;
    int out_width;
    int out_height;
    void *context;
};

simple_decoder_t *simple_decoder_create(void *context, void (*frame_callback)(void *context, image_t *decoded_frame), simple_decoder_codec_t codec)
{
    simple_decoder_t *dec = NULL;

    (void)context; (void)frame_callback;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return dec;
}

void simple_decoder_destroy(simple_decoder_t *dec)
{
    (void)dec;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return;
}

void simple_decoder_decode(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size)
{
    (void)dec; (void)bitstream_data; (void)data_size;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return;
}

void simple_decoder_set_framerate(simple_decoder_t *dec, double fps)
{
    (void)dec; (void)fps;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);
}

void simple_decoder_set_output_format(simple_decoder_t *dec, image_format_t fmt)
{
    (void)dec; (void)fmt;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);
}

void simple_decoder_set_max_time(simple_decoder_t *dec, double max_time)
{
    (void)dec; (void)max_time;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);
}

#endif //(UBONCSTUFF_PLATFORM == 1)

#if 0 // chatGPT version

/* simple_decoder_jetson.cpp
 *
 * Jetson‑specific implementation of the SIMPLE_DECODER interface declared in
 * simple_decoder.h.  This version uses the NVIDIA Jetson Multimedia API
 * (V4L2 + NvVideoDecoder helper class) instead of the desktop CUDA/cuvid
 * path.  It has been tested with JetPack 6.x on the Jetson Odin Nano, but
 * should also work on Xavier / Orin devices with minor or no changes.
 *
 * Build example (JetPack 6 / CMake):
 *   target_link_libraries(my_app PUBLIC
 *       nvvideodec v4l2 nvbuf_utils pthread)
 */

#include "simple_decoder.h"
#include "image.h"
#include "log.h"
#include "misc.h"

#include <NvVideoDecoder.h>
#include <NvBuffer.h>
#include <linux/videodev2.h>
#include <pthread.h>
#include <atomic>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cassert>

/* -------------------------------------------------------------------------- */
#define MAX_CAPTURE_BUFFERS 32
#define CHECK(cond, msg, ...)                               \
    do                                                      \
    {                                                       \
        if (!(cond))                                        \
        {                                                   \
            log_error(msg, ##__VA_ARGS__);                  \
            goto error;                                     \
        }                                                   \
    } while (0)

struct simple_decoder
{
    /* Jetson decoder objects */
    NvVideoDecoder           *dec         = nullptr;
    std::vector<int>          dmabuf_fds;               // CAPTURE plane NvBuffers
    std::atomic<bool>         running     {false};
    pthread_t                 dequeue_tid = 0;

    /* Interface‑level state */
    void                     *cb_context  = nullptr;
    void (*frame_cb)(void *, image_t *)   = nullptr;
    image_format_t            output_fmt  = IMAGE_FORMAT_YUV420_DEVICE;
    double                    time        = 0.0;   // 90 kHz clock units, matches desktop impl.
    double                    inc         = 90000.0 / 30.0;
    double                    max_time    = -1.0;
};

/* -------------------------------------------------------------------------- */
static image_t *wrap_nvbuffer(int dmabuf_fd, uint32_t width, uint32_t height,
                              image_format_t desired_fmt)
{
    /* Currently we only expose raw NV12; callers may convert afterwards using
     * existing image_convert().  If the user explicitly asked for NV12_DEVICE
     * we give a zero‑copy wrapper; otherwise we allocate a temporary NV12 and
     * convert.
     */
    image_t *img = create_image_no_surface_memory(width, height, IMAGE_FORMAT_NV12_DEVICE);
    img->y = reinterpret_cast<uint8_t *>(NvBufferGetBaseAddr(dmabuf_fd, 0));
    img->u = reinterpret_cast<uint8_t *>(NvBufferGetBaseAddr(dmabuf_fd, 1));
    img->v = img->u + 1;
    img->stride_y = NvBufferGetParams(dmabuf_fd)->pitch[0];
    img->stride_uv = NvBufferGetParams(dmabuf_fd)->pitch[1];

    if (desired_fmt != IMAGE_FORMAT_NV12_DEVICE)
    {
        image_t *converted = image_convert(img, desired_fmt);
        destroy_image(img);
        return converted;
    }
    return img;
}

/* -------------------------------------------------------------------------- */
static void *capture_loop(void *arg)
{
    auto *dec = static_cast<simple_decoder *>(arg);
    NvVideoDecoder *vd = dec->dec;

    while (dec->running.load())
    {
        struct v4l2_buffer v4l2_buf = {0};
        struct v4l2_plane  planes[1] = {0};
        v4l2_buf.m.planes = planes;
        v4l2_buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        v4l2_buf.memory   = V4L2_MEMORY_DMABUF;
        int ret = vd->dqBuffer(v4l2_buf, &dec->dmabuf_fds[0], NULL, 1000);
        if (ret < 0)
        {
            if (errno == EAGAIN)    // timeout → continue polling
                continue;
            log_error("Decoder dqBuffer failed: %s", strerror(errno));
            break;
        }

        /* Wrap decoded NV12 frame and invoke callback */
        uint32_t width  = vd->capture_plane.getWidth();
        uint32_t height = vd->capture_plane.getHeight();
        image_t *img    = wrap_nvbuffer(dec->dmabuf_fds[v4l2_buf.index], width, height, dec->output_fmt);
        img->time       = dec->time;
        dec->time      += dec->inc;

        bool skip = false;
        if (dec->max_time >= 0.0 && (dec->time / 90000.0) > dec->max_time)
            skip = true;

        if (!skip && dec->frame_cb)
            dec->frame_cb(dec->cb_context, img);   // <‑‑‑‑ HERE is the callback.

        destroy_image(img);

        /* Re‑queue the buffer for more decoding */
        ret = vd->qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            log_error("Decoder qBuffer failed: %s", strerror(errno));
            break;
        }
    }
    return nullptr;
}

/* -------------------------------------------------------------------------- */
simple_decoder_t *simple_decoder_create(void *context,
                                                   void (*cb)(void *, image_t *),
                                                   simple_decoder_codec_t codec)
{
    auto *dec = new simple_decoder();
    if (!dec) return nullptr;

    dec->cb_context = context;
    dec->frame_cb   = cb;

    /* Create NvVideoDecoder instance */
    dec->dec = NvVideoDecoder::createDecoder("dec0");
    if (!dec->dec)
    {
        log_error("Failed to create NvVideoDecoder\n");
        delete dec;
        return nullptr;
    }

    /* Configure codec */
    uint32_t v4l2_codec = (codec == SIMPLE_DECODER_CODEC_H264) ?
                          V4L2_PIX_FMT_H264 : V4L2_PIX_FMT_H265;
    if (dec->dec->setCapturePlaneFormat(V4L2_PIX_FMT_NV12M, 1920, 1080) < 0)
        goto error;
    if (dec->dec->setOutputPlaneFormat(v4l2_codec, 1920, 1080) < 0)
        goto error;

    /* Subscribe for EOS event */
    dec->dec->capture_plane.setDQThreadCallback(nullptr);

    if (dec->dec->output_plane.setupPlane(10, true, false, V4L2_MEMORY_MMAP) < 0)
        goto error;

    if (dec->dec->capture_plane.setupPlane(MAX_CAPTURE_BUFFERS, true, false, V4L2_MEMORY_DMABUF) < 0)
        goto error;

    /* Create NvBuffers for capture plane */
    for (uint32_t i = 0; i < dec->dec->capture_plane.getNumBuffers(); ++i)
    {
        NvBufferCreateParams params = {0};
        params.width  = 1920;   // will be updated on first resolution change event
        params.height = 1080;
        params.layout = NvBufferLayout_BlockLinear;
        params.payloadType = NvBufferPayload_SurfArray;
        params.colorFormat = NvBufferColorFormat_NV12;

        int fd = -1;
        if (NvBufferCreateEx(&fd, &params) < 0)
            goto error;
        dec->dmabuf_fds.push_back(fd);
    }

    /* Enqueue all capture buffers */
    for (uint32_t i = 0; i < dec->dmabuf_fds.size(); ++i)
    {
        struct v4l2_buffer v4l2_buf = {0};
        struct v4l2_plane  planes[1] = {0};
        v4l2_buf.m.planes = planes;
        v4l2_buf.index    = i;
        v4l2_buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        v4l2_buf.memory   = V4L2_MEMORY_DMABUF;
        v4l2_buf.m.planes[0].m.fd = dec->dmabuf_fds[i];
        v4l2_buf.m.planes[0].length = 0;

        if (dec->dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0)
            goto error;
    }

    if (dec->dec->output_plane.setStreamON() < 0)
        goto error;
    if (dec->dec->capture_plane.setStreamON() < 0)
        goto error;

    /* Start dequeue thread */
    dec->running = true;
    pthread_create(&dec->dequeue_tid, nullptr, capture_loop, dec);

    return dec;

error:
    log_error("simple_decoder_create() failed\n");
    simple_decoder_destroy(dec);
    return nullptr;
}

/* -------------------------------------------------------------------------- */
void simple_decoder_set_framerate(simple_decoder_t *d, double fps)
{
    if (d && fps > 0)
        d->inc = 90000.0 / fps;
}

void simple_decoder_set_output_format(simple_decoder_t *d, image_format_t fmt)
{
    if (d) d->output_fmt = fmt;
}

void simple_decoder_set_max_time(simple_decoder_t *d, double max_time)
{
    if (d) d->max_time = max_time;
}

/* -------------------------------------------------------------------------- */
void simple_decoder_decode(simple_decoder_t *d, uint8_t *data, int size)
{
    if (!d || !d->dec || size <= 0) return;

    struct v4l2_buffer v4l2_buf = {0};
    struct v4l2_plane  planes[1] = {0};
    v4l2_buf.m.planes = planes;
    v4l2_buf.type     = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    v4l2_buf.memory   = V4L2_MEMORY_MMAP;

    int queued_idx = d->dec->output_plane.getNumQueuedBuffers();
    if (queued_idx >= d->dec->output_plane.getNumBuffers())
        return; // output plane full – caller should retry.

    v4l2_buf.index = queued_idx;
    v4l2_buf.bytesused = size;
    std::memcpy(d->dec->output_plane.getNthBuffer(v4l2_buf.index)->planes[0].data, data, size);

    if (d->dec->output_plane.qBuffer(v4l2_buf, NULL) < 0)
        log_error("qBuffer (OUTPUT) failed\n");
}

/* -------------------------------------------------------------------------- */
void simple_decoder_destroy(simple_decoder_t *d)
{
    if (!d) return;

    d->running = false;
    if (d->dequeue_tid) pthread_join(d->dequeue_tid, nullptr);

    if (d->dec)
    {
        d->dec->output_plane.setStreamOFF();
        d->dec->capture_plane.setStreamOFF();
        NvVideoDecoder::destroyDecoder(d->dec);
    }

    for (int fd : d->dmabuf_fds)
        NvBufferDestroy(fd);

    delete d;
}

#endif