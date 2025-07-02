#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <string.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <signal.h>
#include "webcam.h"
#include "jpeg.h"
#include "image.h"


struct webcam
{
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;
    struct v4l2_buffer buf;
    int fd;
    uint64_t first_time;
    uint32_t frames;
    void *buffer;
};

webcam_t *webcam_create(const char *device, int width, int height)
{
    webcam_t *w=(webcam_t *)malloc(sizeof(webcam_t));
    memset(w, 0, sizeof(webcam_t));

    w->fd = open(device, O_RDWR);
    if (w->fd == -1)
    {
        printf("Error opening video device %s\n",device);
        webcam_destroy(w);
        return 0;
    }

    w->fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    w->fmt.fmt.pix.width = width;
    w->fmt.fmt.pix.height = height;
    w->fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    w->fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(w->fd, VIDIOC_S_FMT, &w->fmt) == -1)
    {
        printf("Error setting Pixel Format");
        webcam_destroy(w);
        return 0;
    }

    w->req.count = 1;
    w->req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    w->req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(w->fd, VIDIOC_REQBUFS, &w->req) == -1)
    {
        printf("Error Requesting Buffer");
        webcam_destroy(w);
        return 0;
    }

    w->buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    w->buf.memory = V4L2_MEMORY_MMAP;
    w->buf.index = 0;

    if (ioctl(w->fd, VIDIOC_QUERYBUF, &w->buf) == -1)
    {
        printf("Error Querying Buffer");
        webcam_destroy(w);
        return 0;
    }

    w->buffer = mmap(NULL, w->buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, w->fd, w->buf.m.offset);
    if (w->buffer == MAP_FAILED)
    {
        printf("Error Mapping Buffer");
        webcam_destroy(w);
        return 0;
    }

    if (ioctl(w->fd, VIDIOC_STREAMON, &w->buf.type) == -1)
    {
        printf("Error Starting Capture");
        webcam_destroy(w);
        return 0;
    }

    return w;
}

void webcam_destroy(webcam_t *w)
{
    if (!w) return;

    if (w->fd) close(w->fd);
    free(w);
}

image_t *webcam_capture(webcam_t *w)
{
    w->buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    w->buf.memory = V4L2_MEMORY_MMAP;
    w->buf.index = 0;

    if (ioctl(w->fd, VIDIOC_QBUF, &w->buf) == -1)
    {
        printf("Error Queue Buffer");
        return 0;
    }

    if (ioctl(w->fd, VIDIOC_DQBUF, &w->buf) == -1)
    {
        printf("Error Dequeue Buffer");
        return 0;
    }

    struct timeval tv = w->buf.timestamp;
    uint64_t timestamp90k = (uint64_t)tv.tv_sec * 90000 + (uint64_t)tv.tv_usec * 90 / 1000;

    image_t *img=decode_jpeg((unsigned char *)w->buffer, w->buf.bytesused);
    if (w->frames==0) w->first_time=timestamp90k;
    w->frames++;
    img->time = (timestamp90k-w->first_time)/90000.0;
    return img;
}
