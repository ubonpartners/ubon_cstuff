
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

using namespace std;

#define CHECK_ERROR(cond, str) \
    if(cond) { \
        log_fatal("error %s", str); \
        abort(); \
    }

#define NR_CAPTURE_BUF                          (10)
#define MAX_FIRST_RESCHANGE                     (30)
#define CHUNK_SIZE                              (4 * 1024 * 1024)
#define NR_OUTPUT_BUF_ENCODED_VIDEO_DATA        (2)

typedef struct dma_fd_s {
    int fd;
    int index;
    simple_decoder_t *dec;
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
    uint32_t v4l2_pix_fmt;
    int out_plane_nrbuffer;
    int cap_plane_nrbuffer;
    float fps;
}decctx_cfg_t;

struct simple_decoder
{
    NvVideoDecoder *nvdec;
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

static void nvdec_abort(simple_decoder_t *dec)
{
    log_fatal("aborting");
    dec->got_error = true;
    dec->nvdec->abort();
    abort();
}

static void query_and_set_capture(simple_decoder_t *dec, int from)
{
    int j, ret;
    NvVideoDecoder *nvdec = dec->nvdec;
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
    ret = nvdec->capture_plane.getFormat(format);
    CHECK_ERROR(ret < 0, "Error: Could not get format from decoder capture plane");
    /* Get the display resolution from the decoder.
       Refer ioctl VIDIOC_G_CROP */
    ret = nvdec->capture_plane.getCrop(crop);
    CHECK_ERROR(ret < 0, "Error: Could not get crop from decoder capture plane");
    dec->cfg.dec_width = crop.c.width;
    dec->cfg.dec_height = crop.c.height;
    dec->cfg.window_width = dec->cfg.dec_width;
    dec->cfg.window_height = dec->cfg.dec_height;
    log_info("%s:%d video resolution "
        "dec %d x %d => window %d x %d",
        __func__, __LINE__,
        dec->cfg.dec_width, dec->cfg.dec_height,
        dec->cfg.window_width, dec->cfg.window_height);
    dec->fmt = NVBUF_COLOR_FORMAT_YUV420;
    dec->cfg.v4l2_pix_fmt = V4L2_PIX_FMT_YUV420M;

    /* deinitPlane unmaps the buffers and calls REQBUFS with count 0 */
    nvdec->capture_plane.deinitPlane();
    /* Not necessary to call VIDIOC_S_FMT on decoder capture plane. But
       decoder setCapturePlaneFormat function updates the class variables */
    ret = nvdec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                     format.fmt.pix_mp.width,
                                     format.fmt.pix_mp.height);
    CHECK_ERROR(ret < 0, "Error in setting decoder capture plane format");
    /* Get the min buffers which have to be requested on the capture plane */
    ret = nvdec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
    CHECK_ERROR(ret < 0,
        "Error while getting value of minimum capture plane buffers");
    if(min_dec_capture_buffers > NR_CAPTURE_BUF) {
        log_error("buffer number error %d/%d",min_dec_capture_buffers, NR_CAPTURE_BUF);
        nvdec_abort(dec);
    }
    dec->min_dec_capture_buffers = min_dec_capture_buffers;
    /* Request, Query and export (min + 5) decoder capture plane buffers.
       Refer ioctl VIDIOC_REQBUFS, VIDIOC_QUERYBUF and VIDIOC_EXPBUF */
    int num_buffers=(dec->low_latency) ? min_dec_capture_buffers : (min_dec_capture_buffers + 1);
    ret = nvdec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                       num_buffers, false,
                                       false);
    CHECK_ERROR(ret < 0, "Error in decoder capture plane setup");
    dec->cfg.cap_plane_nrbuffer = nvdec->capture_plane.getNumBuffers();

    /* Start streaming on decoder capture_plane */
    ret = nvdec->capture_plane.setStreamStatus(true);
    CHECK_ERROR(ret < 0, "Error in decoder capture plane streamon");

    /* Enqueue all the empty capture plane buffers */
    for(i = 0; i < (uint32_t)dec->cfg.cap_plane_nrbuffer; i++) {
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        ret = nvdec->capture_plane.qBuffer(v4l2_buf, NULL);
        CHECK_ERROR(ret < 0, "Error Qing buffer at output plane");
    }

    log_info("Resolution change successful: min_dec_capture_buffers %d, num_buffers %d",min_dec_capture_buffers, nvdec->capture_plane.getNumBuffers());
}

static int first_resolution_change(simple_decoder_t *dec)
{
    int j, ret;
    NvVideoDecoder *nvdec = dec->nvdec;
    struct v4l2_event ev;

    for(j = 0; j < MAX_FIRST_RESCHANGE; j++) {
        ret = nvdec->dqEvent(ev, 1000);
        if(ret < 0) {
            if (dec->got_eos)
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

static void process_nvbuffer(simple_decoder_t *dec, NvBuffer *dec_buffer, double time, bool force_skip)
{
    bool skip=force_skip;
    debugf("Process nvbuffer time %f",time);
    if (dec->constraint_min_time_delta!=0)
    {
        float delta=time-dec->last_output_time;
        if ((delta>10.0)||(delta<0))
        {
            if (dec->stats_num_output_surf>0)
            {
                log_error("decoder time constraint unexpected delta %f->%f; restting",dec->last_output_time,time);
            }
            dec->last_output_time=time;
            dec->stats_output_time_reset++;
        }
        else
        {
            skip|=(delta<dec->constraint_min_time_delta);
        }
    }

    if (skip)
    {
        // early return if the frame not needed, avoid a lot of work
        dec->stats_frames_output_skipped++;
        return;
    }

    NvBufSurface *nvbuf_surf = nullptr;
    int ret = NvBufSurfaceFromFd(dec_buffer->planes[0].fd, (void**)(&nvbuf_surf));
    if(ret != 0) {
        log_error("unable to extract NvBufSurfaceFromFd");
        return;
    }

    image_t *dec_img = image_create(dec->cfg.dec_width, dec->cfg.dec_height, IMAGE_FORMAT_NV12_DEVICE);
    ret=nvSurfToImageNV12Device(nvbuf_surf, dec_img, dec_img->stream);
    if (0!=ret)
    {
        log_error("nvSurfToImageYUV420Device failed (%d)",(int)ret);
    }
    dec_img->meta.time = time;
    dec_img->meta.capture_realtime=profile_time();
    dec_img->meta.flags=MD_CAPTURE_REALTIME_SET;

    dec->last_output_time=time;
    // use the frame callback to send the imadddge
    image_sync(dec_img);

    image_t *scaled_out_img=0;
    if (dec->constraint_max_width!=0 && dec->constraint_max_height!=0) {
        determine_scale_size(dec_img->width, dec_img->height,
                            dec->constraint_max_width, dec->constraint_max_width,
                            &dec->scaled_width, &dec->scaled_height,
                            10, 8, 8, false);
        scaled_out_img=image_scale(dec_img, dec->scaled_width, dec->scaled_height);
        debugf("scale to %dx%d",dec->scaled_width, dec->scaled_height);
    }
    else {
        dec->scaled_width=dec_img->width;
        dec->scaled_height=dec_img->height;
        scaled_out_img=image_reference(dec_img);
    }
    dec->stats_num_output_surf++;
    debugf("output surf %d: %f",dec->stats_num_output_surf, scaled_out_img->meta.time);
    dec->frame_callback(dec->context, scaled_out_img);
    image_destroy(scaled_out_img);
    image_destroy(dec_img);
}

static void *dec_capture_loop_fn(void *arg)
{
    cuda_thread_init(); // set context, etc

    int j, ret;
    char thr_name[64];
    simple_decoder_t *dec = (simple_decoder_t *)arg;
    NvVideoDecoder *nvdec = dec->nvdec;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    NvBuffer *dec_buffer;
    struct v4l2_event ev;

    ret = first_resolution_change(dec);
    if(ret < 0) {
        nvdec_abort(dec);
    }
    /* Received the resolution change event, do query_and_set_capture */
    if(!dec->got_error) query_and_set_capture(dec, __LINE__);
    debugf("Dec captureloop");
    /* Exit on error or EOS which is signalled in main() */
    while(!(dec->got_error || nvdec->isInError() || dec->got_eos)) {
        /* Check for resolution change again */
        ret = nvdec->dqEvent(ev, false);
        if(ret == 0) {
            switch(ev.type) {
                case V4L2_EVENT_RESOLUTION_CHANGE:
                    query_and_set_capture(dec, __LINE__);
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
            debugf("dqBuffer");
            ret = nvdec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0);
            debugf("dqBuffer- got buffer!");
            if (ret) {
                // If we've been asked to shut down, just exit the loop.
                if (dec->got_eos) {
                    break;
                }
                // Standard V4L2 “no data” retry
                if (errno == EAGAIN) {
                    debugf("dqBuffer - EAGAIN");
                    usleep(1000);
                    continue;
                }
                // After streamoff(), dqBuffer will return EPIPE; treat that as clean exit.
                if (errno == EPIPE) {
                    break;
                }
                // Otherwise it's a real error
                nvdec_abort(dec);
                log_error("unexpected error on capture_plane.dqBuffer");
                break;
            }
    
            bool force_skip=(v4l2_buf.timestamp.tv_usec&3)!=0;

            double time_in_seconds =(double)v4l2_buf.timestamp.tv_sec +
                                    (double)(v4l2_buf.timestamp.tv_usec&(~3))*0.000001;

            process_nvbuffer(dec, dec_buffer, time_in_seconds, force_skip);

            if (!dec->got_eos)
            {
                if(nvdec->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
                    nvdec_abort(dec);
                    log_error("error while queueing buffer at decoder");
                    break;
                }
            }
            dec->nr_capture++;
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
    NvVideoDecoder *nvdec = NULL;
    simple_decoder_t *dec = NULL;

    debugf("Jetson decoder create");

    dec = (simple_decoder_t *)malloc(sizeof(simple_decoder_t));
    if (dec == 0) return 0;
    memset(dec, 0, sizeof(simple_decoder_t));

    dec->frame_callback = frame_callback;
    dec->context = context;
    dec->low_latency=low_latency;
    dec->decoder_pixfmt = (codec == SIMPLE_DECODER_CODEC_H265) ? V4L2_PIX_FMT_H265 : V4L2_PIX_FMT_H264;
    dec->last_output_time=-5.0;
    dec->nvdec = nvdec = NvVideoDecoder::createVideoDecoder("dec0");
    CHECK_ERROR(!dec, "Could not create the decoder");
    r = nvdec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
    CHECK_ERROR(r < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE");
    r = nvdec->setOutputPlaneFormat(dec->decoder_pixfmt, CHUNK_SIZE);
    CHECK_ERROR(r < 0, "Could not set output plane format");
    r = nvdec->setFrameInputMode(1);
    CHECK_ERROR(r < 0, "Error in decoder setFrameInputMode");
    r = nvdec->disableDPB();
    CHECK_ERROR(r < 0, "Error in decoder disableDPB");
    r = nvdec->setSkipFrames(V4L2_SKIP_FRAMES_TYPE_NONREF);
    CHECK_ERROR(r < 0, "Error in decoder setSkipFrames")
    r = nvdec->output_plane.setupPlane(V4L2_MEMORY_MMAP, NR_OUTPUT_BUF_ENCODED_VIDEO_DATA, true, false);
    CHECK_ERROR(r < 0, "Error while setting up output plane");
    r = nvdec->setMaxPerfMode(1);
    CHECK_ERROR(r < 0, "setMaxPerfMode");
    r = nvdec->output_plane.setStreamStatus(true);
    CHECK_ERROR(r < 0, "Error in output plane stream on");

    dec->cfg.out_plane_nrbuffer = nvdec->output_plane.getNumBuffers();
    log_info("Decoder: DPB disabled, Only decoding reference frames, NR BUF %d",dec->cfg.out_plane_nrbuffer);

    r = pthread_create(&dec->dec_capture_loop, NULL, dec_capture_loop_fn, dec);

    debugf("Jetson decoder created ok");

    return dec;
}

void simple_decoder_destroy(simple_decoder_t *dec)
{
    if (!dec) return;
    log_info("simple decoder destroy");

    // 1) Tell the capture loop to exit
    dec->got_eos = true;

    // 2) Abort decoder to unblock any blocking V4L2 calls
    dec->nvdec->abort();

    // 3) Now wait for the capture thread to finish
    if (dec->dec_capture_loop) {
        // Plain join should now return immediately
        pthread_join(dec->dec_capture_loop, nullptr);
    }

    // 4) Stop and fully deinit the capture plane
    dec->nvdec->capture_plane.setStreamStatus(false);
    dec->nvdec->capture_plane.deinitPlane();

    // 5) Stop the output plane streaming (unblocks any output dqBuffer)
    dec->nvdec->output_plane.setStreamStatus(false);

    // 6) Drain any remaining queued output buffers
    {
        struct v4l2_buffer buf = {};
        struct v4l2_plane planes[MAX_PLANES] = {};
        while (dec->nvdec->output_plane.getNumQueuedBuffers() > 0) {
            buf.m.planes = planes;
            if (dec->nvdec->output_plane.dqBuffer(buf, nullptr, nullptr, 0) < 0)
                break;
        }
    }

    // 7) Delete the decoder object and free our context
    delete dec->nvdec;
    free(dec);

    log_info("decoder destroyed cleanly");
}

void simple_decoder_decode(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size, double frame_time, bool force_skip)
{
    int ret;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    struct timeval tv;
    NvBuffer *buffer;

    dec->stats_num_decode_calls++;
    debugf("Decode %d: t=%f %d bytes",dec->stats_num_decode_calls, frame_time, data_size);

    dec->stats_bytes_decoded+=data_size;
    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, sizeof(planes));
    v4l2_buf.m.planes = planes;

    if(dec->nr_output < dec->cfg.out_plane_nrbuffer) {
        buffer = dec->nvdec->output_plane.getNthBuffer(dec->nr_output);
        v4l2_buf.index = dec->nr_output;
    } else {
        ret = dec->nvdec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
        if(ret < 0) {
            log_error("error dequeuing buffer at output plane");
            nvdec_abort(dec);
            return;
        }
    }
    debugf("Decode has got input buffer");
    memcpy(buffer->planes[0].data, bitstream_data, data_size);
    buffer->planes[0].bytesused = data_size;
    v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
    v4l2_buf.flags= V4L2_BUF_FLAG_TIMESTAMP_COPY;
    tv.tv_sec  = (time_t)frame_time;
    tv.tv_usec = (suseconds_t)((frame_time - tv.tv_sec) * 1e6);
    tv.tv_usec = (tv.tv_usec & (~3)) | (force_skip ? 3 : 0); // ok its a bodge
    v4l2_buf.timestamp = tv;
    ret = dec->nvdec->output_plane.qBuffer(v4l2_buf, NULL);
    if(ret < 0) {
        log_error("error queuing buffer at output plane");
        nvdec_abort(dec);
        return;
    }
    dec->nr_output++;
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
    root["bytes_decoded"]=dec->stats_bytes_decoded;
    root["num_output_surf"]=dec->stats_num_output_surf;
    root["num_decode_calls"]=dec->stats_num_decode_calls;
    root["output_time_reset"]=dec->stats_output_time_reset;
    root["frames_output_skipped"]=dec->stats_frames_output_skipped;
    return root;
}

#endif //(UBONCSTUFF_PLATFORM == 1)
