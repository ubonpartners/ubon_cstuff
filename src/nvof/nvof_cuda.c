#include "log.h"
#include "cuda_stuff.h"
#include "nvof.h"
#include "display.h"
#include <stdio.h>
#include <string.h>
#include <cassert>
#include "nvOpticalFlowCuda.h"
#include <iostream>
#include <mutex>

#if (UBONCSTUFF_PLATFORM == 0) // Desktop Nvidia GPU
#define CHECK_OF(call) \
    do { \
        NV_OF_STATUS _status = (call); \
        if (_status != NV_OF_SUCCESS) { \
            log_fatal("Optical Flow error code %d (%d)", _status, NV_OF_ERR_GENERIC); \
            exit(1); \
        } \
    } while (0)

static std::once_flag initFlag;
static NV_OF_CUDA_API_FUNCTION_LIST nvOFAPI;
static void init_nvof()
{
    CHECK_OF(NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &nvOFAPI));
}

struct nvof
{
    int max_width;
    int max_height;
    int width;
    int height;
    int gridsize;
    int outW;
    int outH;
    uint32_t count=0;
    bool use_nv12;
    CUstream nvof_stream;

    NvOFHandle hOf;
    NvOFGPUBufferHandle inputFrame, referenceFrame, flowBuf, costBuf;
    uint32_t *flowBufHost;
    uint8_t *costBufHost;

    nvof_results_t results;
};

static void create_nvof_buffer(nvof_t *v, int width, int height,
                               NV_OF_BUFFER_USAGE usage,
                               NV_OF_BUFFER_FORMAT fmt,
                               NvOFGPUBufferHandle *handle)
{
    NV_OF_BUFFER_DESCRIPTOR bufDesc = {
        .width = (uint32_t)width,
        .height = (uint32_t)height,
        .bufferUsage = usage,
        .bufferFormat = fmt,
    };
    CHECK_OF(nvOFAPI.nvOFCreateGPUBufferCuda(v->hOf, &bufDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, handle));
}

static void destroy_nvof_buffer(nvof_t *v, NvOFGPUBufferHandle *handle)
{
    if ((*handle)==0) return;
    CHECK_OF(nvOFAPI.nvOFDestroyGPUBufferCuda(*handle));
    *handle=0;
}

static void nvof_set_size(nvof_t *v, int width, int height)
{
    if (v->width==width && v->height==height) return;

    log_trace("NVOF resize to %dx%d nv12:%d",width,height,v->use_nv12);

    destroy_nvof_buffer(v, &v->inputFrame);
    destroy_nvof_buffer(v, &v->referenceFrame);
    destroy_nvof_buffer(v, &v->flowBuf);
    destroy_nvof_buffer(v, &v->costBuf);
    if (v->flowBufHost) cuMemFreeHost(v->flowBufHost);
    if (v->costBufHost) cuMemFreeHost(v->costBufHost);
    v->width=width;
    v->height=height;
    v->gridsize=4;
    v->outW=(width+v->gridsize-1)/v->gridsize;
    v->outH=(height+v->gridsize-1)/v->gridsize;

    NV_OF_INIT_PARAMS initParams;
    memset(&initParams, 0, sizeof(initParams));
    initParams.width=(uint32_t)width;
    initParams.height=(uint32_t)height;
    initParams.perfLevel=NV_OF_PERF_LEVEL_MEDIUM;
    initParams.outGridSize=NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    initParams.mode=NV_OF_MODE_OPTICALFLOW;
    initParams.inputBufferFormat=(v->use_nv12) ?  NV_OF_BUFFER_FORMAT_NV12 : NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    initParams.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
    initParams.enableOutputCost = NV_OF_TRUE;

    if (v->hOf)
    {
        CHECK_OF(nvOFAPI.nvOFDestroy(v->hOf));
        v->hOf=0;
    }
    CHECK_OF(nvOFAPI.nvCreateOpticalFlowCuda(get_CUcontext(), &v->hOf));
    CHECK_OF(nvOFAPI.nvOFInit(v->hOf, &initParams));

    create_nvof_buffer(v, width, height, NV_OF_BUFFER_USAGE_INPUT, initParams.inputBufferFormat, &v->inputFrame);
    create_nvof_buffer(v, width, height, NV_OF_BUFFER_USAGE_INPUT, initParams.inputBufferFormat, &v->referenceFrame);
    create_nvof_buffer(v, v->outW, v->outH, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, &v->flowBuf);
    create_nvof_buffer(v, v->outW, v->outH, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_UINT8, &v->costBuf);
    cuMemAllocHost((void**)&v->flowBufHost, 4*v->outW*v->outH);
    cuMemAllocHost((void**)&v->costBufHost, 1*v->outW*v->outH);

    memset(v->flowBufHost, 0, 4*v->outW*v->outH);
    memset(v->costBufHost, 0, 1*v->outW*v->outH);
    v->count=0;
}

static int get_buf_stride(NvOFGPUBufferHandle buf)
{
    NV_OF_CUDA_BUFFER_STRIDE_INFO si;
    CHECK_OF(nvOFAPI.nvOFGPUBufferGetStrideInfo(buf, &si));
    return si.strideInfo[0].strideXInBytes;
}

nvof_results_t *nvof_execute(nvof_t *v, image_t *img_in)
{
    assert(img_in!=0);

    // decide optimal nvof size

    int w=v->max_width & (~3);
    int h=((w*img_in->height)/img_in->width) & (~3);
    if (h>v->max_height)
    {
        h=v->max_height & (~3);
        w=((h*img_in->width)/img_in->height) & (~3);
    }
    assert(w<= v->max_width && h<= v->max_height);

    nvof_set_size(v, w, h);

    image_t *scaled=image_scale(img_in, v->width, v->height);
    image_format_t target_format=(v->use_nv12) ? IMAGE_FORMAT_NV12_DEVICE : IMAGE_FORMAT_MONO_DEVICE;
    image_t *img=image_convert(scaled, target_format);

    if (v->use_nv12)
    {
        assert(img->stride_y==img->stride_uv);
    }
    //log_debug("nvof size %dx%d img %dx%d",v->width,v->height,img->width,img->height);
    assert(img->width==v->width && img->height==v->height);
    //log_debug("nvof size %dx%d img %dx%d",v->width,v->height,img->width,img->height);
    //display_image("nvof",img);

    NV_OF_EXECUTE_INPUT_PARAMS executeInParams = { 0 };
    NV_OF_EXECUTE_OUTPUT_PARAMS executeOutParams = { 0 };

    executeInParams.inputFrame=v->inputFrame;
    executeInParams.referenceFrame=v->referenceFrame;
    executeInParams.disableTemporalHints=NV_OF_FALSE;
    executeOutParams.outputBuffer = v->flowBuf;
    executeOutParams.outputCostBuffer = v->costBuf;

    CUDA_MEMCPY2D copyP;
    memset(&copyP, 0, sizeof(copyP));
    copyP.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyP.srcDevice = (CUdeviceptr)img->device_mem;
    copyP.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyP.dstDevice = nvOFAPI.nvOFGPUBufferGetCUdeviceptr(v->inputFrame);
    copyP.srcPitch = img->stride_y;
    copyP.dstPitch = get_buf_stride(v->inputFrame);
    copyP.WidthInBytes = img->width;
    copyP.Height = (img->height*(v->use_nv12 ? 3 : 2))/2;

    cuda_stream_add_dependency(v->nvof_stream, img->stream);

    CHECK_CUDA_CALL(cuMemcpy2DAsync(&copyP, v->nvof_stream));
    if (v->count>0)
    {
        CHECK_OF(nvOFAPI.nvOFSetIOCudaStreams(v->hOf, v->nvof_stream, v->nvof_stream));
        CHECK_OF(nvOFAPI.nvOFExecute(v->hOf, &executeInParams, &executeOutParams));
        CUDA_MEMCPY2D copyP1;
        memset(&copyP1, 0, sizeof(copyP1));
        copyP1.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copyP1.srcDevice = nvOFAPI.nvOFGPUBufferGetCUdeviceptr(v->flowBuf);
        copyP1.dstMemoryType = CU_MEMORYTYPE_HOST;
        copyP1.dstHost = v->flowBufHost;
        copyP1.srcPitch = get_buf_stride(v->flowBuf);
        copyP1.dstPitch = 4*v->outW;
        copyP1.WidthInBytes = 4*v->outW;
        copyP1.Height = v->outH;
        CHECK_CUDA_CALL(cuMemcpy2DAsync(&copyP1, v->nvof_stream));

        CUDA_MEMCPY2D copyP2;
        memset(&copyP2, 0, sizeof(copyP2));
        copyP2.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copyP2.srcDevice = nvOFAPI.nvOFGPUBufferGetCUdeviceptr(v->costBuf);
        copyP2.dstMemoryType = CU_MEMORYTYPE_HOST;
        copyP2.dstHost = v->costBufHost;
        copyP2.srcPitch = get_buf_stride(v->costBuf);
        copyP2.dstPitch = v->outW;
        copyP2.WidthInBytes = v->outW;
        copyP2.Height = v->outH;
        CHECK_CUDA_CALL(cuMemcpy2DAsync(&copyP2, v->nvof_stream));

        cuStreamSynchronize(v->nvof_stream); // annoying
    }
    else
    {
        memset(v->costBufHost, 255, v->outW*v->outH); // optical flow not run, set costs to max
        memset(v->flowBufHost, 0, 4*v->outW*v->outH); // optical flow not run, return zero vectors
        cuStreamSynchronize(v->nvof_stream);
    }

    v->count++;

    // switch input and reference buffers

    NvOFGPUBufferHandle temp=v->inputFrame;
    v->inputFrame=v->referenceFrame;
    v->referenceFrame=temp;

    v->results.grid_w=v->outW;
    v->results.grid_h=v->outH;
    v->results.costs=v->costBufHost;
    v->results.flow=(flow_vector_t*)v->flowBufHost;

    destroy_image(scaled);
    destroy_image(img);

    return &v->results;
}

nvof_t *nvof_create(void *context, int max_width, int max_height)
{
    check_cuda_inited();
    std::call_once(initFlag, init_nvof);

    nvof_t *n = (nvof_t *)malloc(sizeof(nvof_t));
    if (n==0) return 0;
    memset(n, 0, sizeof(nvof_t));

    n->use_nv12=true;
    n->max_width=max_width;
    n->max_height=max_height;
    n->nvof_stream=create_cuda_stream();

    return n;
}

void nvof_reset(nvof_t *n)
{
    if (!n) return;
    n->count=0;
}

void nvof_destroy(nvof_t *n)
{
    if (n)
    {
        destroy_cuda_stream(n->nvof_stream);
        destroy_nvof_buffer(n, &n->inputFrame);
        destroy_nvof_buffer(n, &n->referenceFrame);
        destroy_nvof_buffer(n, &n->flowBuf);
        destroy_nvof_buffer(n, &n->costBuf);
        if (n->flowBufHost) cuMemFreeHost(n->flowBufHost);
        if (n->costBufHost) cuMemFreeHost(n->costBufHost);
        if (n->hOf)
        {
            CHECK_OF(nvOFAPI.nvOFDestroy(n->hOf));
        }
        free(n);
    }
}
#endif //(UBONCSTUFF_PLATFORM == 0)
