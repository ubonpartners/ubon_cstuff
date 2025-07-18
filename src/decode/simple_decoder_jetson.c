
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
#include <queue>
#include "cudaEGL.h"
#include "NvUtils.h"
#include "NvVideoDecoder.h"
#include "NvEglRenderer.h"
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include "NvJpegEncoder.h"
#include "NvBufSurface.h"

#include "simple_decoder.h"
#include "cuda_stuff.h"

using namespace std;

#define CHECK_ERROR(cond, str) \
    if(cond) { \
        log_error("%s:%d error %s\n",  __func__, __LINE__, str); \
        fflush(NULL); \
        abort(); \
    }

#define USE_DEC_METADATA     (0)
#define USE_YUV_DUMP         (0)
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
    uint32_t dec_width; /* decoded video stream */
    uint32_t dec_height; /* decoded video stream */
    uint32_t window_width; /* video window */
    uint32_t window_height; /* video window */
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

typedef struct q_ctx_s {
    pthread_mutex_t q_mutex; // queue mutex
    std::queue < dma_fd_t * > *mq; // empty queue
    std::queue < dma_fd_t * > *fq; // filled queue
    int32_t mq_len; // empty queue length
    int32_t fq_len; // filled queue length
}q_ctx_t;

struct simple_decoder
{
    int is_h265;
    int do_jpeg;
    int do_dump_yuv;
    NvVideoDecoder *dec;
    NvJPEGEncoder *jenc;
    decctx_cfg_t cfg;
    uint32_t decoder_pixfmt;
    int32_t min_dec_capture_buffers;
    NvBufSurfaceColorFormat fmt;

    dma_fd_t dma_fd[NR_CAPTURE_BUF];
    q_ctx_t *q_ctx;

    int nr_capture; // number of frames from capture plane
    int nr_output; // number of buffers enqueued to output plane
    pthread_t dec_capture_loop;
    bool got_error;
    bool got_eos;

    int width;
    int height;
    int out_width;
    int out_height;

    image_format_t output_format;
    double time;
    double time_increment;
    double max_time;
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
    log_info("%s:%d %s\n",  __func__, __LINE__, b);

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

/**
  * Print plane param for debug purpose
  */
void print_plane_param(simple_decoder_t *ctx, NvBufSurface *nvbuf_surf)
{
    int index = 0, l = 0;
    char b[1024];
    NvBufSurfaceParams *surface_list = &nvbuf_surf->surfaceList[index];
    NvBufSurfacePlaneParams *pp = &nvbuf_surf->surfaceList->planeParams;
    NvBufSurfaceMappedAddr *maddr = &nvbuf_surf->surfaceList->mappedAddr;

    //return; /* uncomment to use */

    l += snprintf(b + l, sizeof(b) - l,
        "[%5d] [0: %d %d %d %d %d %d] [1: %d %d %d %d %d %d] [2: %d %d %d %d %d %d] ",
        ctx->nr_capture,
        pp->width[0], pp->height[0], pp->pitch[0], pp->bytesPerPix[0],
        pp->offset[0], pp->psize[0],
        pp->width[1], pp->height[1], pp->pitch[1], pp->bytesPerPix[1],
        pp->offset[1], pp->psize[1],
        pp->width[2], pp->height[2], pp->pitch[2], pp->bytesPerPix[2],
        pp->offset[2], pp->psize[2]);
    l += snprintf(b + l, sizeof(b) - l,
        "  [w %u h %u p %u fmt 0x%x] ",
        surface_list->width, surface_list->height, surface_list->pitch,
        surface_list->colorFormat);
    if(surface_list->colorFormat == NVBUF_COLOR_FORMAT_NV12) {
        l += snprintf(b + l, sizeof(b) - l, "NV12 ");
    }
    if(surface_list->colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
        l += snprintf(b + l, sizeof(b) - l, "RGBA ");
    }
    if(surface_list->colorFormat == NVBUF_COLOR_FORMAT_YUV420) {
        l += snprintf(b + l, sizeof(b) - l, "YUV420 ");
    }
    log_info("\n%s", b);

    return;
}

static int copy_dmabuf(simple_decoder_t *ctx, NvBufSurface *nvbuf_surf,
    uint8_t *out, int plane)
{
    int index = 0, width, height, pitch, byte_per_pix, len;
    uint8_t *p;
    uint8_t *src;
    uint8_t *dest = out;
    NvBufSurfacePlaneParams *pp = &nvbuf_surf->surfaceList->planeParams;
    NvBufSurfaceMappedAddr *maddr = &nvbuf_surf->surfaceList->mappedAddr;

    if(NvBufSurfaceMap(nvbuf_surf, index, plane, NVBUF_MAP_READ_WRITE) != 0) {
        log_error("%s:%d Failed to map NvBufSurface\n", __func__, __LINE__);
        nvdec_abort_ctx(ctx);
    }

    if(NvBufSurfaceSyncForCpu(nvbuf_surf, index, plane) != 0) {
        log_error("%s:%d Failed to sync surface for CPU\n");
        nvdec_abort_ctx(ctx);
    }
    width = pp->width[plane];
    height = pp->height[plane];
    pitch = pp->pitch[plane];
    byte_per_pix = pp->bytesPerPix[plane];
    src = (uint8_t *)nvbuf_surf->surfaceList->mappedAddr.addr[plane];
    len = width * byte_per_pix;

    for (int i = 0; i < height; i++) {
        memcpy(dest, src, len);
        src += pitch;
        dest += len;
    }

    if(NvBufSurfaceUnMap(nvbuf_surf, index, plane) < 0) {
        log_error("%s:%d Failed to unmap NvBufSurface\n", __func__, __LINE__);
        nvdec_abort_ctx(ctx);
    }
    len = (int)(dest - out);


    return len;
}

static int cuda_memcpy_image(simple_decoder_t *ctx, image_t *dec_img, CUeglFrame *eglFrame)
{
    void* y_ptr = eglFrame->frame.pPitch[0]; // Y plane
    int y_size = ctx->cfg.dec_width *ctx->cfg.dec_height;

    cudaError_t y_result = cudaMemcpy(
        dec_img->y, // Destination
        y_ptr, // Source
        y_size, // Size to copy
        cudaMemcpyDeviceToHost // GPU to Host
    );
    if (y_result != cudaSuccess) {
        fprintf(stderr, "Y plane copy failed: %s\n", cudaGetErrorString(y_result));
    }

    return y_size;
}

int create_image_forward(simple_decoder_t *ctx, NvBufSurface *nvbuf_surf, CUeglFrame *eglFrame)
{
    int do_malloc = 0, do_free = 0, do_memset = 1;
    int index = 0, w, h, ret, stride_y, stride_uv;
    NvBufSurfaceParams *surface_list = &nvbuf_surf->surfaceList[index];
    NvBufSurfacePlaneParams *pp = &nvbuf_surf->surfaceList->planeParams;
    image_t *dec_img;
    char fname[256];
    FILE *fp = NULL;

    if(surface_list->colorFormat != NVBUF_COLOR_FORMAT_YUV420) {
        log_error("%s:%d Invalid color format %d\n",
            __func__, __LINE__, surface_list->colorFormat);
        nvdec_abort_ctx(ctx);
    }
    w = ctx->cfg.dec_width;
    h = ctx->cfg.dec_height;
    stride_y = pp->pitch[0];
    stride_uv = pp->pitch[1];

    dec_img = create_image(w, h, IMAGE_FORMAT_YUV420_HOST);
    /*
    dec_img = create_image_no_surface_memory(w, h, IMAGE_FORMAT_YUV420_HOST);
    */
    if(dec_img->width != w || dec_img->height != h) {
        log_error("%s:%d Size mismatch - NvBufSurface : image struct\n",
            __func__, __LINE__, surface_list->colorFormat);
        nvdec_abort_ctx(ctx);
    }
    // allocate the yuv buffers
    if(do_malloc) {
        dec_img->y = (uint8_t *)malloc(stride_y * h);
        dec_img->u = (uint8_t *)malloc(stride_uv * (h / 2));
        dec_img->v = (uint8_t *)malloc(stride_uv * (h / 2));
        dec_img->stride_y = stride_y;
        dec_img->stride_uv = stride_uv;
    }

    // memset the yuv buffers
    if(do_memset) {
        memset(dec_img->y, 0, (dec_img->stride_y * h));
        memset(dec_img->u, 0x80, dec_img->stride_uv * (h / 2));
        memset(dec_img->v, 0x80, dec_img->stride_uv * (h / 2));
    }

    //ret = cuda_memcpy_image(ctx, dec_img, eglFrame);
    if(ctx->do_dump_yuv) {
        sprintf(fname, "image_%06d.yuv", ctx->nr_capture);
        fp = fopen(fname, "wb");
        if(!fp) nvdec_abort_ctx(ctx);
        log_info("%s:%d yuv file %s\n", __func__, __LINE__, fname);
    }

    // copy each of the yuv planes to the corresponding buffers
    ret = copy_dmabuf(ctx, nvbuf_surf, dec_img->y, 0);
    if(fp) fwrite(dec_img->y, 1, ret, fp);
    ret = copy_dmabuf(ctx, nvbuf_surf, dec_img->u, 1);
    if(fp) fwrite(dec_img->u, 1, ret, fp);
    ret = copy_dmabuf(ctx, nvbuf_surf, dec_img->v, 2);
    if(fp) fwrite(dec_img->v, 1, ret, fp);
    if(fp) fclose(fp);

    dec_img->time = ctx->time;
    ctx->time += ctx->time_increment;
    // use the frame callback to send the image
    ctx->frame_callback(ctx->context, dec_img);

    // free the yuv buffers
    if(do_free) {
        free(dec_img->y);
        free(dec_img->u);
        free(dec_img->v);
        dec_img->y = dec_img->u = dec_img->v = NULL;
    }
    destroy_image(dec_img);
    (void)ret;

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

static int register_new_dma_fd(simple_decoder_t *ctx, dma_fd_t *dma_fd)
{
    int qlen;
    q_ctx_t *q_ctx = ctx->q_ctx;

    pthread_mutex_lock(&q_ctx->q_mutex);
    do {
        q_ctx->mq->push(dma_fd);
        q_ctx->mq_len++;
        qlen = q_ctx->mq_len;
    } while(0);
    pthread_mutex_unlock(&q_ctx->q_mutex);

    return qlen;
}

static dma_fd_t *get_dma_fd2(simple_decoder_t *ctx, q_ctx_t *q_ctx, int *type)
{
    dma_fd_t *dma_fd;

    if(q_ctx->mq_len > 0) {
        dma_fd = q_ctx->mq->front();
        q_ctx->mq->pop();
        q_ctx->mq_len--;
        *type = 0x01;
        return dma_fd;
    }
    /* return NULL; */
    if(q_ctx->fq_len > 0) {
        dma_fd = q_ctx->fq->front();
        q_ctx->fq->pop();
        q_ctx->fq_len--;
        *type = 0x02;
        return dma_fd;
    }

    log_error("%s:%d both queue empty %d %d\n",
        __func__, __LINE__, q_ctx->mq_len, q_ctx->fq_len);
    nvdec_abort_ctx(ctx);

    return NULL;
}

static dma_fd_t *get_dma_fd(simple_decoder_t *ctx)
{
    int type = 0;
    dma_fd_t *dma_fd = NULL;
    q_ctx_t *q_ctx = ctx->q_ctx;

    pthread_mutex_lock(&q_ctx->q_mutex);
    do {
        dma_fd = get_dma_fd2(ctx, q_ctx, &type);
    } while(0);
    pthread_mutex_unlock(&q_ctx->q_mutex);

    if(dma_fd) {
    } else {
        log_error("%s:%d NO DMA FD [m%2d f%2d type %d]\n",
            __func__, __LINE__, q_ctx->mq_len, q_ctx->fq_len, type);
    }

    return dma_fd;
}

static int queue_dma_fd(simple_decoder_t *ctx, dma_fd_t *dma_fd)
{
    q_ctx_t *q_ctx = ctx->q_ctx;

    pthread_mutex_lock(&q_ctx->q_mutex);
    do {
        gettimeofday(&dma_fd->tv, NULL);
        q_ctx->fq->push(dma_fd);
        q_ctx->fq_len++;
    } while(0);
    pthread_mutex_unlock(&q_ctx->q_mutex);

    return 0;
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
    qlen = register_new_dma_fd(ctx, dma_fd);
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
    if(ctx->cfg.tr_width == 0)  ctx->cfg.tr_width = ctx->cfg.dec_width;
    if(ctx->cfg.tr_height == 0)  ctx->cfg.tr_height = ctx->cfg.dec_height;
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

    log_info("%s:%d Resolution change successful at %ld.%06ld\n",
        __func__, __LINE__, tv.tv_sec, tv.tv_usec);
    log_info("%s:%d min_dec_capture_buffers %d, num_buffers %d\n",
        __func__, __LINE__,
        min_dec_capture_buffers, dec->capture_plane.getNumBuffers());

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
    CUeglFrame *eglFrame;
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
    eglFrame = &dma_fd->eglFramePtr;

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
    if((USE_DEC_METADATA) || (ctx->nr_capture < 2)) {
        print_plane_param(ctx, nvbuf_surf);
    }
    create_image_forward(ctx, nvbuf_surf, eglFrame);
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
            ctx->nr_capture++;
        }
    }

    return NULL;
}

simple_decoder_t *simple_decoder_create(void *context, void (*frame_callback)(void *context, image_t *decoded_frame), simple_decoder_codec_t codec)
{
    int j, r;
    NvVideoDecoder *dec = NULL;
    simple_decoder_t *ctx = NULL;
    q_ctx_t *q_ctx = NULL;

    ctx = (simple_decoder_t *)malloc(sizeof(simple_decoder_t));
    if (ctx == 0) return 0;
    memset(ctx, 0, sizeof(simple_decoder_t));
    q_ctx = new q_ctx_t;
    ctx->q_ctx = q_ctx;

    ctx->frame_callback = frame_callback;
    ctx->context = context;

    if(codec == SIMPLE_DECODER_CODEC_H265) {
        ctx->is_h265 = 1;
    }
    ctx->decoder_pixfmt = find_v4l2_pixfmt(ctx->is_h265);
    for(j = 0; j < NR_CAPTURE_BUF; j++) {
        ctx->dma_fd[j].fd = -1;
    }

    pthread_mutex_init(&q_ctx->q_mutex, NULL);
    q_ctx->mq = new queue <dma_fd_t *>;
    q_ctx->fq = new queue <dma_fd_t *>;
    q_ctx->mq_len = 0;
    q_ctx->fq_len = 0;
    ctx->do_jpeg = 0;
    ctx->do_dump_yuv = USE_YUV_DUMP;

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
    /* Set V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT control to false
       so that application can send chunks of encoded data instead of forming
       complete frames. This needs to be done before setting format on the
       output plane. */
    r = dec->setFrameInputMode(1);
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
    if(ctx->do_jpeg) {
    }

    /* use YUV420 or RGBA as the output color format */
    ctx->cfg.out_pixfmt = 1; /* NV12 */
    ctx->cfg.out_pixfmt = 3; /* RGBA */
    ctx->cfg.out_pixfmt = 2; /* YUV420 */

    log_info("%s:%d creating thread", __func__, __LINE__);
    r = pthread_create(&ctx->dec_capture_loop, NULL, dec_capture_loop_fn, ctx);

    return ctx;
}

void simple_decoder_destroy(simple_decoder_t *dec)
{
    int ret = 0;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    simple_decoder_t *ctx = dec;
    q_ctx_t *q_ctx = ctx->q_ctx;

    if(ctx == NULL) return;

    log_info("%s:%d stopping", __func__, __LINE__);
    /* As EOS, dequeue all the output planes */
    while(ctx->dec->output_plane.getNumQueuedBuffers() > 0 &&
           !ctx->got_error && !ctx->dec->isInError()) {
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;
        ret = ctx->dec->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
        if(ret < 0) {
            log_error("%s:%d error dequeuing buffer at output plane\n",
                __func__, __LINE__);
            nvdec_abort_ctx(ctx);
            break;
        }
    }

    /* Mark EOS for the decoder capture thread */
    ctx->got_eos = true;

    sleep(1);
    pthread_cancel(ctx->dec_capture_loop);
    sleep(1);

    delete q_ctx->mq;
    delete q_ctx->fq;
    delete ctx->q_ctx;

    if(ctx->dec_capture_loop) pthread_join(ctx->dec_capture_loop, NULL);
    free(ctx);

    return;
}

void simple_decoder_decode(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size)
{
    int ret;
    simple_decoder_t *ctx = dec;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    struct timeval tv;
    NvBuffer *buffer;
    v4l2_ctrl_videodec_inputbuf_metadata d;

    do {
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.m.planes = planes;
        gettimeofday(&tv, NULL);

        if(ctx->nr_output < ctx->cfg.out_plane_nrbuffer) {
            buffer = ctx->dec->output_plane.getNthBuffer(ctx->nr_output);
            v4l2_buf.index = ctx->nr_output;
        } else {
            ret = ctx->dec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
            if(ret < 0) {
                log_error("%s:%d error dequeuing buffer at output plane\n",
                    __func__, __LINE__);
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
        /* NVDEC currently only preserves timestamp, so
         * using the timestamp filed to send the buffer index
         * v4l2_buf timestamp as buffer index */
        v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
        v4l2_buf.timestamp = tv;

        /* Queue an empty buffer to signal EOS to the decoder
           i.e. set v4l2_buf.m.planes[0].bytesused = 0 and queue the buffer */
        ret = ctx->dec->output_plane.qBuffer(v4l2_buf, NULL);
        if(ret < 0) {
            log_error("%s:%d error queuing buffer at output plane\n",
                __func__, __LINE__);
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

void simple_decoder_set_framerate(simple_decoder_t *dec, double fps)
{
    log_info("%s:%d fps = %.3f\n", __func__, __LINE__, fps);
    if (dec && fps > 0) dec->time_increment = 1.0 / fps;
}

void simple_decoder_set_output_format(simple_decoder_t *dec, image_format_t fmt)
{
    log_info("%s:%d fmt = %d\n", __func__, __LINE__, fmt);
    if (dec) dec->output_format = fmt;
}

void simple_decoder_set_max_time(simple_decoder_t *dec, double max_time)
{
    log_info("%s:%d max_time = %.3f\n", __func__, __LINE__, max_time);
    if (dec) dec->max_time = max_time;
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
