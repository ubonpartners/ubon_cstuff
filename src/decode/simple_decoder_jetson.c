
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
#include "NvEglRenderer.h"
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include "NvBufSurface.h"

#include "simple_decoder.h"
#include "cuda_stuff.h"

#define CHECK_ERROR(cond, str) \
    if(cond) { \
        log_error("%s:%d error %s\n",  __func__, __LINE__, str); \
        fflush(NULL); \
        abort(); \
    }

#define USE_DEC_METADATA     (0)
#define NR_CAPTURE_BUF       (10)
#define MAX_FIRST_RESCHANGE  (30)
#define CHUNK_SIZE           (4 * 1024 * 1024)
#define NR_OUTPUT_BUF        (10)

typedef struct dma_fd_s {
    int fd;
    int index;
    simple_decoder_t *ctx;
    struct timeval tv;
    CUeglFrame eglFramePtr;
    CUgraphicsResource pResource;
    EGLImageKHR egl_image;
}dma_fd_t;

typedef struct decctx_cfg_s {
    int disable_rendering;
    int fullscreen;
    int do_jpeg;
    uint32_t dec_width; /* decoded video stream */
    uint32_t dec_height; /* decoded video stream */
    uint32_t window_width; /* video window */
    uint32_t window_height; /* video window */
    uint32_t net_width; /* inference network */
    uint32_t net_height; /* inference network */
    uint32_t net_channel; /* inference network */
    uint32_t tr_width; /* transform output */
    uint32_t tr_height; /* transform output */
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
    dma_fd_t dma_fd[NR_CAPTURE_BUF];
    pthread_t dec_capture_loop;
    int do_exit;
    bool got_error;
    bool got_eos;

    int width;
    int height;
    int out_width;
    int out_height;
    void *context;
    void (*frame_callback)(void *context, image_t *decoded_frame);
};

static void nvdec_abort_ctx(simple_decoder_t *ctx)
{
    log_error("%s:%d aborting\n", __func__, __LINE__);
    ctx->got_error = true;
    ctx->dec->abort();
    abort();
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
                d->CodecParams.H264DecParams.dpbInfo.nActiveRefFrames);
        }

        if (d->FrameDecStats.DecodeError)
        {
            /* decoder error status metadata. */
            dec_stats = &d->FrameDecStats;
            l += snprintf(b + l, sizeof(b) - l, "ErrorType=%d Decoded MBs=%d Concealed MBs=%d",
                dec_stats->DecodeError, dec_stats->DecodedMBs, dec_stats->ConcealedMBs);
        }
    }
    log_info("%s:%d %s\n",  __func__, __LINE__, b);

    return 0;
}

static uint32_t find_v4l2_pixfmt(int is_h265)
{
    uint32_t pixfmt = 0;

    if(is_h265) {
        pixfmt = V4L2_PIX_FMT_H265;
    } else {
        pixfmt = V4L2_PIX_FMT_H264;
    }

    return pixfmt;
}

static int alloc_dma_bufsurface(int index, simple_decoder_t *ctx,
    dma_fd_t *dma_fd, int width, int height)
{
    int ret, qlen;
    NvBufSurf::NvCommonAllocateParams params;

    /* Create PitchLinear output buffer for transform. */
    params.memType = NVBUF_MEM_SURFACE_ARRAY;
    params.width = width;
    params.height = height;
    params.layout = NVBUF_LAYOUT_PITCH;
    params.colorFormat = ctx->fmt;
    params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;
    ret = NvBufSurf::NvAllocate(&params, 1, &dma_fd->fd);
    gettimeofday(&dma_fd->tv, NULL);
    log_info("%s:%d dst_dma_fd[%02d:%02d] %3d / %2d\n",
        __func__, __LINE__, 0, index, dma_fd->fd, qlen);

    return ret;
}

static int free_dma_bufsurface(int index, simple_decoder_t *ctx, int *dma_fd)
{
    int ret;

    if(*dma_fd == -1)  {
        return 0;
    }
    ret = NvBufSurf::NvDestroy(*dma_fd);
    if(ret < 0) nvdec_abort_ctx(ctx);
    log_info("%s:%d dst_dma_fd[%02d] %d\n", __func__, __LINE__,
        0, *dma_fd);
    *dma_fd = -1;
    (void)ctx;

    return ret;
}

static dma_fd_t *get_dma_fd(simple_decoder_t *ctx)
{
    dma_fd_t *dma_fd = NULL;

    return dma_fd;
}

static int queue_dma_fd(simple_decoder_t *ctx, dma_fd_t *dma_fd)
{

    return 0;
}

static void query_and_set_capture(simple_decoder_t *ctx, int from)
{
    int j, ret;
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_format format;
    struct v4l2_crop crop;
    uint32_t i;
    int32_t min_dec_capture_buffers;
    dma_fd_t *dma_fd;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    NvBufSurface *nvbuf_surf = NULL;
    CUresult status;

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
        "dec %d x %d => transform %d x %d => window %d x %d\n",
        __func__, __LINE__,
        ctx->cfg.dec_width, ctx->cfg.dec_height,
        ctx->cfg.tr_width, ctx->cfg.tr_height,
        ctx->cfg.window_width, ctx->cfg.window_height);

    switch(ctx->cfg.out_pixfmt) {
        case 1:
            ctx->fmt = NVBUF_COLOR_FORMAT_NV12;
            ctx->cfg.v4l2_pix_fmt = V4L2_PIX_FMT_NV12;
            break;
        case 2:
            ctx->fmt = NVBUF_COLOR_FORMAT_YUV420;
            ctx->cfg.v4l2_pix_fmt = V4L2_PIX_FMT_YUV420M;
            break;
        case 3:
            ctx->fmt = NVBUF_COLOR_FORMAT_RGBA;
            ctx->cfg.v4l2_pix_fmt = V4L2_PIX_FMT_RGBA32;
            break;
        case 4:
            ctx->fmt = NVBUF_COLOR_FORMAT_NV16;
            ctx->cfg.v4l2_pix_fmt = V4L2_PIX_FMT_NV16M;
            break;
        case 5:
            ctx->fmt = NVBUF_COLOR_FORMAT_NV24;
            ctx->cfg.v4l2_pix_fmt = V4L2_PIX_FMT_NV24;
            break;
        default:
            ctx->fmt = NVBUF_COLOR_FORMAT_INVALID;
            ctx->cfg.v4l2_pix_fmt = 0;
            break;
    }
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
        log_error("%s:%d %d %d buffer number error\n",
            __func__, __LINE__, min_dec_capture_buffers, NR_CAPTURE_BUF);
        nvdec_abort_ctx(ctx);
    }
    ctx->min_dec_capture_buffers = min_dec_capture_buffers;
    /* Request, Query and export (min + 5) decoder capture plane buffers.
       Refer ioctl VIDIOC_REQBUFS, VIDIOC_QUERYBUF and VIDIOC_EXPBUF */
    ret = dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                       min_dec_capture_buffers + 5, false,
                                       false);
    CHECK_ERROR(ret < 0, "Error in decoder capture plane setup");
    ctx->cfg.cap_plane_nrbuffer = dec->capture_plane.getNumBuffers();

    for(j = 0; j < min_dec_capture_buffers; j++) {
        ctx->dma_fd[j].index = j;
        ctx->dma_fd[j].ctx = ctx;
        /* ctx->dma_fd[j].instance = ctx->instance; */
    }

    for(j = 0; j < min_dec_capture_buffers; j++) {
        ret = free_dma_bufsurface(j, ctx, &ctx->dma_fd[j].fd);
        CHECK_ERROR(ret < 0, "Error: Error in BufferDestroy");
    }
    for(j = 0; j < min_dec_capture_buffers; j++) {
        dma_fd = &ctx->dma_fd[j];
        ret = alloc_dma_bufsurface(j, ctx, dma_fd,
            ctx->cfg.tr_width, ctx->cfg.tr_height);
        CHECK_ERROR(ret == -1, "create dmabuf failed");

        /* fd to egl frameptr mapping */
        ret = NvBufSurfaceFromFd(dma_fd->fd, (void **)&nvbuf_surf);
        if(ret != 0) {
            log_error("%s:%d unable to extract NvBufSurfaceFromFd\n",
                __func__, __LINE__);
            CHECK_ERROR(ret < 0, "unable to extract NvBufSurfaceFromFd");
            nvdec_abort_ctx(ctx);
        }
        if(nvbuf_surf->surfaceList[0].mappedAddr.eglImage == NULL) {
            if(NvBufSurfaceMapEglImage(nvbuf_surf, 0) != 0) {
                log_error("%s:%d Error while mapping "
                    "dmabuf fd (0x%X) to EGLImage\n",
                    __func__, __LINE__, dma_fd->fd);
                nvdec_abort_ctx(ctx);
                return;
            }
        }
        dma_fd->egl_image = nvbuf_surf->surfaceList[0].mappedAddr.eglImage;
        if(dma_fd->egl_image == NULL) {
            log_error("%s:%d Error while mapping "
                "dmabuf fd (0x%X) to EGLImage\n",
                __func__, __LINE__, dma_fd->fd);
            nvdec_abort_ctx(ctx);
            return;
        }
        dma_fd->pResource = NULL;
        cudaFree(0);

        status = cuGraphicsEGLRegisterImage(&dma_fd->pResource,
           dma_fd->egl_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        if(status != CUDA_SUCCESS) {
            log_error("%s:%d cuGraphicsEGLRegisterImage failed: %d %d, "
                "cuda process stop\n", __func__, __LINE__, dma_fd->fd, status);
            nvdec_abort_ctx(ctx);
        }
        status = cuGraphicsResourceGetMappedEglFrame(&dma_fd->eglFramePtr,
            dma_fd->pResource, 0, 0);
        if(status != CUDA_SUCCESS) {
            log_error("%s:%d cuGraphicsResourceGetMappedEglFrame "
                "failed: %d %d\n", __func__, __LINE__, dma_fd->fd, status);
            nvdec_abort_ctx(ctx);
        }
    }

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
            if(errno == EAGAIN) {
                log_error("%s:%d [%d %d] Timeout "
                    "V4L2_EVENT_RESOLUTION_CHANGE\n", __func__, __LINE__,
                    j, ret);
                ret = -1;
            } else {
                log_error("%s:%d error in dequeueing decoder event\n",
                    __func__, __LINE__);
                ret = -2;
                break;
            }
        }
        if((ret == 0) && (ev.type == V4L2_EVENT_RESOLUTION_CHANGE)) {
            log_info("%s:%d rx first V4L2_EVENT_RESOLUTION_CHANGE\n",
                __func__, __LINE__);
            return 0;
        }
    }

    return ret;
}

static int process_nvbuffer(simple_decoder_t *ctx, NvBuffer *dec_buffer)
{
    int ret, dst_dma_fd;
    dma_fd_t *dma_fd;
    NvBufSurface *nvbuf_surf = NULL;
    EGLImageKHR egl_image = NULL;
    int use_dynamic_egl = 0;

    if(ctx->min_dec_capture_buffers == -1) nvdec_abort_ctx(ctx);
    dma_fd = get_dma_fd(ctx);
    if(dma_fd == NULL) nvdec_abort_ctx(ctx);
    if(dma_fd == NULL) return 0;
    dst_dma_fd = dma_fd->fd;

    /* Clip & Stitch can be done by adjusting rectangle. */
    NvBufSurf::NvCommonTransformParams transform_params;
    transform_params.src_top = 0;
    transform_params.src_left = 0;
    transform_params.src_width = ctx->cfg.dec_width;
    transform_params.src_height = ctx->cfg.dec_height;
    transform_params.dst_top = 0;
    transform_params.dst_left = 0;
    transform_params.dst_width = ctx->cfg.tr_width;
    transform_params.dst_height = ctx->cfg.tr_height;
    transform_params.flag = NVBUFSURF_TRANSFORM_FILTER;
    transform_params.flip = NvBufSurfTransform_None;
    transform_params.filter = NvBufSurfTransformInter_Nearest;

    /* Perform Blocklinear to PitchLinear conversion. */
    ret = NvBufSurf::NvTransform(&transform_params,
        dec_buffer->planes[0].fd, dst_dma_fd);
    if(ret == -1) {
        log_error("%s:%d transform failed\n", __func__, __LINE__);
        return -1;
    }

    queue_dma_fd(ctx, dma_fd);

    /* Get EGLImage from dmabuf fd */
    ret = NvBufSurfaceFromFd(dst_dma_fd, (void**)(&nvbuf_surf));
    if(ret != 0) {
        log_error("%s:%d unable to extract NvBufSurfaceFromFd\n",
            __func__, __LINE__);
        return -1;
    }
    if(ctx->cfg.out_pixfmt == 2) {
    }
    if(use_dynamic_egl) {
        if(NvBufSurfaceMapEglImage(nvbuf_surf, 0) != 0) {
            log_error("%s:%d unable to map EGL Image\n",
                    __func__, __LINE__);
            return -1;
        }
        egl_image = nvbuf_surf->surfaceList[0].mappedAddr.eglImage;
        if(egl_image == NULL) {
            log_error("%s:%d Error while mapping dmabuf fd (0x%X) "
                "to EGLImage\n", __func__, __LINE__, dst_dma_fd);
            return -1;
        }
    } else {
        /* take the calculated egl_image during initialization */
        egl_image = dma_fd->egl_image;
    }

    /* Map EGLImage to CUDA buffer, and call CUDA kernel to
       draw a 32x32 pixels black box on left-top of each frame */
    /*
    HandleEGLImage(&egl_image);
    */

    /* Destroy EGLImage */
    if(use_dynamic_egl) {
        if(NvBufSurfaceUnMapEglImage(nvbuf_surf, 0) != 0) {
            log_error("%s:%d unable to unmap EGL Image\n",
                __func__, __LINE__);
            return -1;
        }
    }
    egl_image = NULL;

    return 0;
}

static void *dec_capture_loop_fn(void *arg)
{
    int j, ret;
    char thr_name[64];
    simple_decoder_t *ctx = (simple_decoder_t *)arg;
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    NvBuffer *dec_buffer;
    struct v4l2_event ev;
    struct timeval tv;
    v4l2_ctrl_videodec_outputbuf_metadata d;

    snprintf(thr_name, sizeof(thr_name), "dec_cap_%p", ctx->context);
    prctl(PR_SET_NAME, thr_name, 0, 0, 0);

    ret = first_resolution_change(ctx);
    if(ret < 0) {
        nvdec_abort_ctx(ctx);
    }
    /* Received the resolution change event, do query_and_set_capture */
    if(!ctx->got_error) query_and_set_capture(ctx, __LINE__);

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
        gettimeofday(&tv, NULL);

        /* Decoder capture loop */
        while(1) {
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));
            v4l2_buf.m.planes = planes;
            /* Dequeue a valid capture_plane buffer containing YUV BL data */
            ret = dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0);
            if(ret) {
                if(errno == EAGAIN) {
                    usleep(1000);
                } else {
                    nvdec_abort_ctx(ctx);
                    log_info("%s:%d error while calling "
                        "dequeue at capture plane\n", __func__, __LINE__);
                }
                break;
            }
            if(USE_DEC_METADATA) {
                ret = dec->getMetadata(v4l2_buf.index, d);
                print_metadata(ctx, &d);
            }
            ret = process_nvbuffer(ctx, dec_buffer);
            if(ret == -1) break;
            /* If not writing to file,
             * Queue the buffer back once it has been used. */
            if(dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
                nvdec_abort_ctx(ctx);
                log_error("%s:%d error while queueing buffer at decoder"
                    "capture plane\n", __func__, __LINE__);
                break;
            }
        }
    }

/*
    for(j = 0; ; j++) {
        log_info("%s:%d j %6d\n", __func__, __LINE__, j);
        if(ctx->do_exit) {
            log_info("%s:%d exiting\n", __func__, __LINE__);
            break;
        }
        usleep(10 * 1000);
    }
*/

    return NULL;
}

simple_decoder_t *simple_decoder_create(void *context, void (*frame_callback)(void *context, image_t *decoded_frame), simple_decoder_codec_t codec)
{
    int j, r;
    NvVideoDecoder *dec = NULL;
    simple_decoder_t *ctx = NULL;

    ctx = (simple_decoder_t *)malloc(sizeof(simple_decoder_t));
    if (ctx == 0) return 0;
    memset(ctx, 0, sizeof(simple_decoder_t));

    ctx->frame_callback = frame_callback;
    ctx->context = context;

    if(codec == SIMPLE_DECODER_CODEC_H265) {
        ctx->is_h265 = 1;
    }
    ctx->decoder_pixfmt = find_v4l2_pixfmt(ctx->is_h265);
    for(j = 0; j < NR_CAPTURE_BUF; j++) {
        ctx->dma_fd[j].fd = -1;
    }

    dec = NvVideoDecoder::createVideoDecoder("dec0");
    CHECK_ERROR(!dec, "Could not create the decoder");
    log_info("%s:%d ctx = %p dec = %p\n",  __func__, __LINE__, ctx, dec); \
    ctx->dec = dec;
    r = dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
    CHECK_ERROR(r < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE");
    /* Set the max size of the outputPlane buffers, here is
       CHUNK_SIZE, which contains the encoded data in bytes */
    r = dec->setOutputPlaneFormat(ctx->decoder_pixfmt, CHUNK_SIZE);
    CHECK_ERROR(r < 0, "Could not set output plane format");
    r = dec->setFrameInputMode(0);
    CHECK_ERROR(r < 0, "Error in decoder setFrameInputMode");
    /* Request MMAP buffers for writing encoded video data */
    r = dec->output_plane.setupPlane(V4L2_MEMORY_MMAP,
        NR_OUTPUT_BUF, true, false);
    CHECK_ERROR(r < 0, "Error while setting up output plane");
    ctx->cfg.out_plane_nrbuffer = dec->output_plane.getNumBuffers();
    log_info("%s:%d NR_BUFFER %d\n", __func__, __LINE__, ctx->cfg.out_plane_nrbuffer);
    if(USE_DEC_METADATA) {
        r = dec->enableMetadataReporting();
        CHECK_ERROR(r < 0, "Error metadata reporting");
    }
    /* Start streaming on decoder output_plane */
    r = dec->output_plane.setStreamStatus(true);
    CHECK_ERROR(r < 0, "Error in output plane stream on");

    ctx->cfg.out_pixfmt = 2; /* YUV420 */
    ctx->cfg.out_pixfmt = 3; /* RGBA */

    log_info("%s:%d creating thread", __func__, __LINE__);
    r = pthread_create(&ctx->dec_capture_loop, NULL, dec_capture_loop_fn, ctx);

    return ctx;
}

void simple_decoder_destroy(simple_decoder_t *dec)
{
    if(dec == NULL) return;

    log_info("%s:%d stopping", __func__, __LINE__);
    dec->do_exit = 1;
    if(dec->dec_capture_loop) pthread_join(dec->dec_capture_loop, NULL);
    free(dec);

    return;
}

void simple_decoder_decode(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size)
{
    (void)dec; (void)bitstream_data; (void)data_size;
    //log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

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
