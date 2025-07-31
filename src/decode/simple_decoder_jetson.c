
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

#define debugf if (0) log_info

using namespace std;

#define CHECK_ERROR(cond, str) \
    if(cond) { \
        log_fatal("error %s", str); \
        abort(); \
    }

#define USE_DEC_METADATA                        (0)
#define NR_CAPTURE_BUF                          (10)
#define MAX_FIRST_RESCHANGE                     (30)
#define CHUNK_SIZE                              (4 * 1024 * 1024)
#define NR_OUTPUT_BUF_ENCODED_VIDEO_DATA        (3)

typedef struct dma_fd_s {
    int fd;
    int index;
    simple_decoder_t *ctx;
    struct timeval tv;
}dma_fd_t;

typedef struct decctx_cfg_s {
    int disable_rendering;
    int fullscreen;
    uint32_t dec_width; /* decoded video stream */
    uint32_t dec_height; /* decoded video stream */
    uint32_t window_width; /* video window */
    uint32_t window_height; /* video window */
    uint32_t window_x;
    uint32_t window_y;
    uint32_t out_pixfmt;
    uint32_t v4l2_pix_fmt;
    int out_plane_nrbuffer;
    int cap_plane_nrbuffer;
    int32_t is_h265;
    float fps;
}decctx_cfg_t;

struct simple_decoder
{
    int is_h265;
    NvVideoDecoder *dec;
    decctx_cfg_t cfg;
    uint32_t decoder_pixfmt;
    int32_t min_dec_capture_buffers;
    NvBufSurfaceColorFormat fmt;

    int nr_capture; // number of frames from capture plane
    int nr_output; // number of buffers enqueued to output plane
    pthread_t dec_capture_loop;
    bool got_error;
    bool got_eos;
    bool low_latency;

    int width;
    int height;
    int out_width;
    int out_height;

    image_format_t output_format;
    void *context;
    void (*frame_callback)(void *context, image_t *decoded_frame);

    int constraint_max_width;
    int constraint_max_height;
    double constraint_min_time_delta;
    int scaled_width, scaled_height;
    double last_output_time;

    uint32_t stats_num_output_surf;
    uint32_t stats_num_decode_calls;
    uint32_t stats_output_time_reset;
    uint32_t stats_frames_output_skipped;
    uint64_t stats_bytes_decoded;
};

static void nvdec_abort_ctx(simple_decoder_t *ctx)
{
    log_fatal("aborting");
    ctx->got_error = true;
    ctx->dec->abort();
    abort();
}

static int print_input_metadata(simple_decoder_t *ctx,
    v4l2_ctrl_videodec_inputbuf_metadata *d)
{
    int l = 0;
    char b[1024];
    uint32_t frame_num;

    frame_num = ctx->dec->output_plane.getTotalDequeuedBuffers() - 1;
    l += snprintf(b + l, sizeof(b) - l, "Frame: [%d] ", frame_num);

    if (d->nBitStreamError & V4L2_DEC_ERROR_SPS) {
        l += snprintf(b + l, sizeof(b) - l, "ERROR_SPS ");
    } else if (d->nBitStreamError & V4L2_DEC_ERROR_PPS) {
        l += snprintf(b + l, sizeof(b) - l, "ERROR_PPS ");
    } else if (d->nBitStreamError & V4L2_DEC_ERROR_SLICE_HDR) {
        l += snprintf(b + l, sizeof(b) - l, "ERROR_SLICE_HDR ");
    } else if (d->nBitStreamError & V4L2_DEC_ERROR_MISSING_REF_FRAME) {
        l += snprintf(b + l, sizeof(b) - l, "ERROR_MISSING_REF_FRAME ");
    } else if (d->nBitStreamError & V4L2_DEC_ERROR_VPS) {
        l += snprintf(b + l, sizeof(b) - l, "ERROR_VPS ");
    } else {
        l += snprintf(b + l, sizeof(b) - l, "ERROR_NONE ");
    }
    log_info("%s", b);

    return 0;
}


static int print_metadata(simple_decoder_t *ctx,
    v4l2_ctrl_videodec_outputbuf_metadata *d)
{
    int l = 0;
    char b[1024];
    uint32_t frame_num;
    v4l2_ctrl_videodec_statusmetadata *dec_stats;

    frame_num = ctx->dec->capture_plane.getTotalDequeuedBuffers() - 1;
    l += snprintf(b + l, sizeof(b) - l, "[%d] Frame: [%d] ", 0, frame_num);

    if(d->bValidFrameStatus) {
        if(ctx->decoder_pixfmt == V4L2_PIX_FMT_H264) {
            l += snprintf(b + l, sizeof(b) - l, "H264 ");
            /* metadata for H264 input stream. */
            switch(d->CodecParams.H264DecParams.FrameType) {
                case 0:
                    l += snprintf(b + l, sizeof(b) - l, "FrameType = B ");
                    break;
                case 1:
                    l += snprintf(b + l, sizeof(b) - l, "FrameType = P ");
                    break;
                case 2:
                    l += snprintf(b + l, sizeof(b) - l, "FrameType = I ");
                    if(d->CodecParams.H264DecParams.dpbInfo.currentFrame.bIdrFrame) {
                        l += snprintf(b + l, sizeof(b) - l, "(IDR) ");
                    }
                    break;
            }
            l += snprintf(b + l, sizeof(b) - l, "nActiveRefFrames = %d ",
                d->CodecParams.H264DecParams.dpbInfo.nActiveRefFrames);
        }

        if(ctx->decoder_pixfmt == V4L2_PIX_FMT_H265) {
            l += snprintf(b + l, sizeof(b) - l, "H265 ");
            /* metadata for HEVC input stream. */
            switch(d->CodecParams.HEVCDecParams.FrameType) {
                case 0:
                    l += snprintf(b + l, sizeof(b) - l, "FrameType = B ");
                    break;
                case 1:
                    l += snprintf(b + l, sizeof(b) - l, "FrameType = P ");
                    break;
                case 2:
                    l += snprintf(b + l, sizeof(b) - l, "FrameType = I ");
                    if(d->CodecParams.HEVCDecParams.dpbInfo.currentFrame.bIdrFrame) {
                        l += snprintf(b + l, sizeof(b) - l, "(IDR) ");
                    }
                    break;
            }
            l += snprintf(b + l, sizeof(b) - l, "nActiveRefFrames = %d ",
                d->CodecParams.HEVCDecParams.dpbInfo.nActiveRefFrames);
        }

        if (d->FrameDecStats.DecodeError)
        {
            /* decoder error status metadata. */
            dec_stats = &d->FrameDecStats;
            l += snprintf(b + l, sizeof(b) - l, "ErrorType=%d Decoded MBs=%d Concealed MBs=%d",
                dec_stats->DecodeError, dec_stats->DecodedMBs, dec_stats->ConcealedMBs);
        }
    }
    log_info("%s", b);

    return 0;
}

static void query_and_set_capture(simple_decoder_t *ctx, int from)
{
    int j, ret;
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_format format;
    struct v4l2_crop crop;
    struct timeval tv;
    uint32_t i;
    int32_t min_dec_capture_buffers;
    dma_fd_t *dma_fd;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    NvBufSurface *nvbuf_surf = NULL;
    CUresult status;

    gettimeofday(&tv, NULL);
    /* Get capture plane format from the decoder.
       This may change after resolution change event.
       Refer ioctl VIDIOC_G_FMT */
    ret = dec->capture_plane.getFormat(format);
    CHECK_ERROR(ret < 0, "Error: Could not get format from decoder capture plane");
    /* Get the display resolution from the decoder.
       Refer ioctl VIDIOC_G_CROP */
    ret = dec->capture_plane.getCrop(crop);
    CHECK_ERROR(ret < 0, "Error: Could not get crop from decoder capture plane");
    ctx->cfg.dec_width = crop.c.width;
    ctx->cfg.dec_height = crop.c.height;
    ctx->cfg.window_width = ctx->cfg.dec_width;
    ctx->cfg.window_height = ctx->cfg.dec_height;
    log_info("%s:%d video resolution "
        "dec %d x %d => window %d x %d",
        __func__, __LINE__,
        ctx->cfg.dec_width, ctx->cfg.dec_height,
        ctx->cfg.window_width, ctx->cfg.window_height);
    ctx->fmt = NVBUF_COLOR_FORMAT_YUV420;
    ctx->cfg.v4l2_pix_fmt = V4L2_PIX_FMT_YUV420M;

    /* deinitPlane unmaps the buffers and calls REQBUFS with count 0 */
    dec->capture_plane.deinitPlane();
    /* Not necessary to call VIDIOC_S_FMT on decoder capture plane. But
       decoder setCapturePlaneFormat function updates the class variables */
    ret = dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                     format.fmt.pix_mp.width,
                                     format.fmt.pix_mp.height);
    CHECK_ERROR(ret < 0, "Error in setting decoder capture plane format");
    /* Get the min buffers which have to be requested on the capture plane */
    ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
    CHECK_ERROR(ret < 0,
        "Error while getting value of minimum capture plane buffers");
    if(min_dec_capture_buffers > NR_CAPTURE_BUF) {
        log_error("buffer number error %d/%d",min_dec_capture_buffers, NR_CAPTURE_BUF);
        nvdec_abort_ctx(ctx);
    }
    ctx->min_dec_capture_buffers = min_dec_capture_buffers;
    /* Request, Query and export (min + 5) decoder capture plane buffers.
       Refer ioctl VIDIOC_REQBUFS, VIDIOC_QUERYBUF and VIDIOC_EXPBUF */
    ret = dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                       min_dec_capture_buffers + 1, false,
                                       false);
    CHECK_ERROR(ret < 0, "Error in decoder capture plane setup");
    ctx->cfg.cap_plane_nrbuffer = dec->capture_plane.getNumBuffers();

    /* Start streaming on decoder capture_plane */
    ret = dec->capture_plane.setStreamStatus(true);
    CHECK_ERROR(ret < 0, "Error in decoder capture plane streamon");

    /* Enqueue all the empty capture plane buffers */
    for(i = 0; i < (uint32_t)ctx->cfg.cap_plane_nrbuffer; i++) {
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        ret = dec->capture_plane.qBuffer(v4l2_buf, NULL);
        CHECK_ERROR(ret < 0, "Error Qing buffer at output plane");
    }

    log_info("Resolution change successful");
    log_info("min_dec_capture_buffers %d, num_buffers %d",min_dec_capture_buffers, dec->capture_plane.getNumBuffers());

    return;
}

static int first_resolution_change(simple_decoder_t *ctx)
{
    int j, ret;
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_event ev;

    for(j = 0; j < MAX_FIRST_RESCHANGE; j++) {
        ret = dec->dqEvent(ev, 1000);
        if(ret < 0) {
            if (ctx->got_eos)
            {
                log_error("Timeout V4L2_EVENT_RESOLUTION_CHANGE with EOS set");
                return 0;
            }
            if(errno == EAGAIN) {
                log_error("Timeout V4L2_EVENT_RESOLUTION_CHANGE");
                ret = -1;
            } else {
                log_error("error in dequeueing decoder event");
                ret = -2;
                break;
            }
        }
        if((ret == 0) && (ev.type == V4L2_EVENT_RESOLUTION_CHANGE)) {
            log_info("x first V4L2_EVENT_RESOLUTION_CHANGE");
            return 0;
        }
    }

    return ret;
}

extern int nvSurfToImageNV12Device(NvBufSurface *nvSurf,
                              image_t      *img,
                              CUstream      stream );

static void process_nvbuffer(simple_decoder_t *ctx, NvBuffer *dec_buffer, double time)
{
    bool skip=false;
    debugf("Process nvbuffer time %f",time);
    if (ctx->constraint_min_time_delta!=0)
    {
        float delta=time-ctx->last_output_time;
        if ((delta>10.0)||(delta<0))
        {
            if (ctx->stats_num_output_surf>0)
            {
                log_error("decoder time constraint unexpected delta %f->%f; restting",ctx->last_output_time,time);
            }
            ctx->last_output_time=time;
            ctx->stats_output_time_reset++;
        }
        else
        {
            skip|=(delta<ctx->constraint_min_time_delta);
        }
    }

    if (skip)
    {
        // early return if the frame not needed, avoid a lot of work
        ctx->stats_frames_output_skipped++;
        return;
    }

    NvBufSurface *nvbuf_surf = nullptr;
    int ret = NvBufSurfaceFromFd(dec_buffer->planes[0].fd, (void**)(&nvbuf_surf));
    if(ret != 0) {
        log_error("unable to extract NvBufSurfaceFromFd");
        return;
    }

    image_t *dec_img = image_create(ctx->cfg.dec_width, ctx->cfg.dec_height, IMAGE_FORMAT_NV12_DEVICE);
    ret=nvSurfToImageNV12Device(nvbuf_surf, dec_img, dec_img->stream);
    if (0!=ret)
    {
        log_error("nvSurfToImageYUV420Device failed (%d)",(int)ret);
    }
    dec_img->meta.time = time;
    dec_img->meta.capture_realtime=profile_time();
    dec_img->meta.flags=MD_CAPTURE_REALTIME_SET;

    ctx->last_output_time=time;
    // use the frame callback to send the imadddge
    image_sync(dec_img);

    image_t *scaled_out_img=0;
    if (ctx->constraint_max_width!=0 && ctx->constraint_max_height!=0) {
        determine_scale_size(dec_img->width, dec_img->height,
                            ctx->constraint_max_width, ctx->constraint_max_width,
                            &ctx->scaled_width, &ctx->scaled_height,
                            10, 8, 8, false);
        scaled_out_img=image_scale(dec_img, ctx->scaled_width, ctx->scaled_height);
        debugf("scale to %dx%d",ctx->scaled_width, ctx->scaled_height);
    }
    else {
        ctx->scaled_width=dec_img->width;
        ctx->scaled_height=dec_img->height;
        scaled_out_img=image_reference(dec_img);
    }
    ctx->stats_num_output_surf++;
    debugf("output surf %d: %f",ctx->stats_num_output_surf, scaled_out_img->meta.time);
    ctx->frame_callback(ctx->context, scaled_out_img);
    image_destroy(scaled_out_img);
    image_destroy(dec_img);
}

static void *dec_capture_loop_fn(void *arg)
{
    cuda_thread_init(); // set context, etc

    int j, ret;
    char thr_name[64];
    simple_decoder_t *ctx = (simple_decoder_t *)arg;
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    NvBuffer *dec_buffer;
    struct v4l2_event ev;
    //struct timeval tv;
    v4l2_ctrl_videodec_outputbuf_metadata d;

    snprintf(thr_name, sizeof(thr_name), "dec_cap_%p", ctx->context);
    prctl(PR_SET_NAME, thr_name, 0, 0, 0);

    ret = first_resolution_change(ctx);
    if(ret < 0) {
        nvdec_abort_ctx(ctx);
    }
    /* Received the resolution change event, do query_and_set_capture */
    if(!ctx->got_error) query_and_set_capture(ctx, __LINE__);
    debugf("Dec captureloop");
    /* Exit on error or EOS which is signalled in main() */
    while(!(ctx->got_error || dec->isInError() || ctx->got_eos)) {
        /* Check for resolution change again */
        ret = dec->dqEvent(ev, false);
        if(ret == 0) {
            switch(ev.type) {
                case V4L2_EVENT_RESOLUTION_CHANGE:
                    query_and_set_capture(ctx, __LINE__);
                    continue;
            }
        }
        //gettimeofday(&tv, NULL);

        /* Decoder capture loop */
        while(1) {
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));
            v4l2_buf.m.planes = planes;

            /* Dequeue a valid capture_plane buffer containing YUV BL data */
            ret = dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0);
            if (ret) {
                // If we've been asked to shut down, just exit the loop.
                if (ctx->got_eos) {
                    break;
                }
                // Standard V4L2 “no data” retry
                if (errno == EAGAIN) {
                    usleep(1000);
                    continue;
                }
                // After streamoff(), dqBuffer will return EPIPE; treat that as clean exit.
                if (errno == EPIPE) {
                    break;
                }
                // Otherwise it's a real error
                nvdec_abort_ctx(ctx);
                log_error("unexpected error on capture_plane.dqBuffer");
                break;
            }

            //tv.tv_sec  = (time_t)t;
            //tv.tv_usec = (suseconds_t)((t - buf.timestamp.tv_sec) * 1e6);
            //printf("2tv_sec = %ld, tv_usec = %ld\n",
            //    (long)tv.tv_sec, (long)tv.tv_usec);

            double time_in_seconds =(double)v4l2_buf.timestamp.tv_sec +
                                    (double)v4l2_buf.timestamp.tv_usec*0.000001;

            if(USE_DEC_METADATA) {
                ret = dec->getMetadata(v4l2_buf.index, d);
                print_metadata(ctx, &d);
            }
            process_nvbuffer(ctx, dec_buffer, time_in_seconds);
            /* If not writing to file,
             * Queue the buffer back once it has been used. */
            if (!ctx->got_eos)
            {
                if(dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
                    nvdec_abort_ctx(ctx);
                    log_error("error while queueing buffer at decoder");
                    break;
                }
            }
            ctx->nr_capture++;
        }
    }
    return NULL;
}

simple_decoder_t *simple_decoder_create(void *context,
                                        void (*frame_callback)(void *context, image_t *decoded_frame),
                                        simple_decoder_codec_t codec,
                                        bool low_latency)
{
    int j, r;
    NvVideoDecoder *dec = NULL;
    simple_decoder_t *ctx = NULL;

    debugf("Jetson decoder create");

    ctx = (simple_decoder_t *)malloc(sizeof(simple_decoder_t));
    if (ctx == 0) return 0;
    memset(ctx, 0, sizeof(simple_decoder_t));

    ctx->frame_callback = frame_callback;
    ctx->context = context;
    ctx->low_latency=low_latency;

    if(codec == SIMPLE_DECODER_CODEC_H265) {
        ctx->is_h265 = 1;
    }
    ctx->decoder_pixfmt = (ctx->is_h265) ? V4L2_PIX_FMT_H265 : V4L2_PIX_FMT_H264;
    ctx->last_output_time=-5.0;
    dec = NvVideoDecoder::createVideoDecoder("dec0");
    CHECK_ERROR(!dec, "Could not create the decoder");
    log_trace("ctx = %p dec = %p", ctx, dec);
    ctx->dec = dec;
    r = dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
    CHECK_ERROR(r < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE");
    /* Set the max size of the outputPlane buffers, here is
       CHUNK_SIZE, which contains the encoded data in bytes */
    r = dec->setOutputPlaneFormat(ctx->decoder_pixfmt, CHUNK_SIZE);
    CHECK_ERROR(r < 0, "Could not set output plane format");
    /* Set V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT control to false
       so that application can send chunks of encoded data instead of forming
       complete frames. This needs to be done before setting format on the
       output plane. */
    r = dec->setFrameInputMode(0);
    CHECK_ERROR(r < 0, "Error in decoder setFrameInputMode");

    int ret = dec->disableDPB();
    if (ret < 0)
        log_error("Failed to set V4L2_CID_MPEG_VIDEO_DISABLE_DPB");
    else
        log_info("Disabled DPB for lower latency");

    dec->setSkipFrames(V4L2_SKIP_FRAMES_TYPE_NONREF);
    log_info("Only decoding reference frames (B/dispP skipped)");


    /* Request MMAP buffers for writing encoded video data */
    r = dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, NR_OUTPUT_BUF_ENCODED_VIDEO_DATA, true, false);
    CHECK_ERROR(r < 0, "Error while setting up output plane");
    ctx->cfg.out_plane_nrbuffer = dec->output_plane.getNumBuffers();
    log_trace("NR_BUFFER %d",ctx->cfg.out_plane_nrbuffer);
    if(USE_DEC_METADATA) {
        r = dec->enableMetadataReporting();
        CHECK_ERROR(r < 0, "Error metadata reporting");
    }
    r = dec->setMaxPerfMode(1);
    CHECK_ERROR(r < 0, "setMaxPerfMode");
    /* Start streaming on decoder output_plane */
    r = dec->output_plane.setStreamStatus(true);
    CHECK_ERROR(r < 0, "Error in output plane stream on");

    ctx->cfg.out_pixfmt = 2; /* YUV420 */

    log_info("%s:%d creating thread", __func__, __LINE__);
    r = pthread_create(&ctx->dec_capture_loop, NULL, dec_capture_loop_fn, ctx);

    debugf("Jetson decoder created ok");

    return ctx;
}

void simple_decoder_destroy(simple_decoder_t *ctx)
{
    if (!ctx) return;
    log_info("simple decoder destroy");

    // 1) Tell the capture loop to exit
    ctx->got_eos = true;

    // 2) Abort decoder to unblock any blocking V4L2 calls
    ctx->dec->abort();

    // 3) Now wait for the capture thread to finish
    if (ctx->dec_capture_loop) {
        // Plain join should now return immediately
        pthread_join(ctx->dec_capture_loop, nullptr);
    }

    // 4) Stop and fully deinit the capture plane
    ctx->dec->capture_plane.setStreamStatus(false);
    ctx->dec->capture_plane.deinitPlane();

    // 5) Stop the output plane streaming (unblocks any output dqBuffer)
    ctx->dec->output_plane.setStreamStatus(false);

    // 6) Drain any remaining queued output buffers
    {
        struct v4l2_buffer buf = {};
        struct v4l2_plane planes[MAX_PLANES] = {};
        while (ctx->dec->output_plane.getNumQueuedBuffers() > 0) {
            buf.m.planes = planes;
            if (ctx->dec->output_plane.dqBuffer(buf, nullptr, nullptr, 0) < 0)
                break;
        }
    }

    // 7) Delete the decoder object and free our context
    delete ctx->dec;
    free(ctx);

    log_info("decoder destroyed cleanly");
}

void simple_decoder_decode(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size, double frame_time)
{
    int ret;
    simple_decoder_t *ctx = dec;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    struct timeval tv;
    NvBuffer *buffer;
    v4l2_ctrl_videodec_inputbuf_metadata d;

    dec->stats_num_decode_calls++;
    debugf("Decode %d: %f",dec->stats_num_decode_calls, frame_time);

    dec->stats_bytes_decoded+=data_size;
    do {
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.m.planes = planes;

        if(ctx->nr_output < ctx->cfg.out_plane_nrbuffer) {
            buffer = ctx->dec->output_plane.getNthBuffer(ctx->nr_output);
            v4l2_buf.index = ctx->nr_output;
        } else {
            ret = ctx->dec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
            if(ret < 0) {
                log_error("error dequeuing buffer at output plane");
                nvdec_abort_ctx(ctx);
                break;
            }
        }
        if((USE_DEC_METADATA) && (v4l2_buf.flags & V4L2_BUF_FLAG_ERROR)) {
            ret = ctx->dec->getInputMetadata(v4l2_buf.index, d);
            print_input_metadata(ctx, &d);
        }
        memcpy(buffer->planes[0].data, bitstream_data, data_size);
        buffer->planes[0].bytesused = data_size;
        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
        /* NVDEC currently only preserves timestamp*/
        v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
        tv.tv_sec  = (time_t)frame_time;
        tv.tv_usec = (suseconds_t)((frame_time - tv.tv_sec) * 1e6);
        v4l2_buf.timestamp = tv;

        /* Queue an empty buffer to signal EOS to the decoder
           i.e. set v4l2_buf.m.planes[0].bytesused = 0 and queue the buffer */
        ret = ctx->dec->output_plane.qBuffer(v4l2_buf, NULL);
        if(ret < 0) {
            log_error("error queuing buffer at output plane");
            nvdec_abort_ctx(ctx);
            break;
        }
        if(v4l2_buf.m.planes[0].bytesused == 0) {
            log_info("%s:%d input completed\n", __func__, __LINE__);
            break;
        }
        if((USE_DEC_METADATA) && (v4l2_buf.flags & V4L2_BUF_FLAG_ERROR)) {
            ret = ctx->dec->getInputMetadata(v4l2_buf.index, d);
            print_input_metadata(ctx, &d);
        }
        ctx->nr_output++;
    } while(0);
    return;
}


void simple_decoder_set_output_format(simple_decoder_t *dec, image_format_t fmt)
{
    log_info("%s:%d fmt = %d", __func__, __LINE__, fmt);
    if (dec) dec->output_format = fmt;
}

void simple_decoder_constrain_output(simple_decoder_t *dec, int max_width, int max_height, double min_time_delta)
{
    dec->constraint_max_width=max_width;
    dec->constraint_max_height=max_height;
    dec->constraint_min_time_delta=min_time_delta;
}

YAML::Node simple_decoder_get_stats(simple_decoder *dec)
{
    YAML::Node root;
    root["stats_bytes_decoded"]=dec->stats_bytes_decoded;
    return root;
}

#endif //(UBONCSTUFF_PLATFORM == 1)
