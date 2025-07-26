#include <vpi/Types.h>
#include <vpi/Stream.h>
#include <vpi/algo/OpticalFlowDense.h>
#include "vpi.h"
#include <string.h>
#include <stdio.h>
#include "cuda_stuff.h"

#define CHECK_STATUS_VPI(STMT)                                \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            printf("VPI fail: %s\n",buffer);                  \
        }                                                     \
    } while (0);

struct vpi_of
{
    int width,height;
    image_t *prev;
    VPIStream stream;
    int32_t grid_sizes[1];
    VPIPayload payload;
};

vpi_of_t *vpi_of_create(void *context, int width, int height)
{
    check_cuda_inited();
    vpi_of_t *v = (vpi_of_t *)malloc(sizeof(vpi_of_t));
    if (v==0) return 0;
    memset(v, 0, sizeof(vpi_of_t));
    v->width=width;
    v->height=height;
    v->grid_sizes[0]=4;
    CHECK_STATUS_VPI(vpiStreamCreate(VPI_BACKEND_OFA|VPI_BACKEND_CUDA|VPI_BACKEND_NVENC, &v->stream));
    CHECK_STATUS_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA, width, height, VPI_IMAGE_FORMAT_NV12_BL,
                              &v->grid_sizes[0], 1,
                              VPI_OPTICAL_FLOW_QUALITY_HIGH, &v->payload));
    return v;
}

static void vpi_wrap_luma_plane(image_t *img, VPIImageData *d)
{   
    memset(d, 0, sizeof(VPIImageData));
    d->bufferType=VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    d->buffer.pitch.format = VPI_IMAGE_FORMAT_Y8;
    d->buffer.pitch.numPlanes = 1;
    d->buffer.pitch.planes[0].width = img->width;
    d->buffer.pitch.planes[0].height = img->height;
    d->buffer.pitch.planes[0].pitchBytes = img->stride_y;
    d->buffer.pitch.planes[0].data = (void*)img->device_mem;
}

void vpi_of_execute(vpi_of_t *v, image_t *img)
{
    image_t *converted=image_convert(img, IMAGE_FORMAT_YUV420_DEVICE);
    assert(converted!=0);
    image_t *converted_scaled=image_scale(converted, v->width, v->height);
    assert(converted_scaled!=0);
    destroy_image(converted);

    if (v->prev==0)
    {
        v->prev=converted_scaled;
        return;
    }

    VPIImageData yPlaneData0;
    VPIImageData yPlaneData1;
    VPIImage img0_gray = NULL;
    VPIImage img1_gray = NULL;
    VPIPyramid pyr0 = NULL, pyr1 = NULL;
    VPIPayload pyrOptFlow = NULL;
    VPIArray flow = NULL;
    
    vpi_wrap_luma_plane(v->prev, &yPlaneData0);
    vpi_wrap_luma_plane(converted_scaled, &yPlaneData1);

    vpiImageSetWrapper(img0_gray, &yPlaneData0);
    vpiImageSetWrapper(img1_gray, &yPlaneData1);


    // Wait for all operations to finish
    vpiStreamSync(v->stream);

    destroy_image(v->prev);
    v->prev=converted_scaled;
    
}

void vpi_of_destroy(vpi_of_t *v)
{
    if (v)
    {
        if (v->prev) destroy_image(v->prev);
        free(v);
    }
}
