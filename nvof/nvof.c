#include "cuda_stuff.h"
#include "nvof.h"
#include <stdio.h>
#include <string.h>
#include <cassert>
#include "nvOpticalFlowCuda.h"
#include <iostream>
#include <mutex>

#define CHECK_CUDA(call) \
    if ((call) != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA Error at %s:%d %s\n", __FILE__, __LINE__); exit(1); \
    }

#define CHECK_OF(call) \
    do { \
        NV_OF_STATUS _status = (call); \
        if (_status != NV_OF_SUCCESS) { \
            fprintf(stderr, "Optical Flow error at %s:%d - code %d (bad ptr is %d)\n", __FILE__, __LINE__, _status,NV_OF_ERR_INVALID_PTR); \
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
    int width;
    int height;
    int gridsize;
    int outW;
    int outH;
    uint32_t count=0;

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

    destroy_nvof_buffer(v, &v->inputFrame);
    destroy_nvof_buffer(v, &v->referenceFrame);
    destroy_nvof_buffer(v, &v->flowBuf);
    destroy_nvof_buffer(v, &v->costBuf);
    if (v->flowBufHost) cuMemFreeHost(v->flowBufHost);
    if (v->costBufHost) cuMemFreeHost(v->costBufHost);
    v->width=width;
    v->height=height;
    v->gridsize=4;
    v->outW=width/v->gridsize;
    v->outH=height/v->gridsize;

    NV_OF_INIT_PARAMS initParams;
    memset(&initParams, 0, sizeof(initParams));
    initParams.width=(uint32_t)width;
    initParams.height=(uint32_t)height;
    initParams.perfLevel=NV_OF_PERF_LEVEL_MEDIUM;
    initParams.outGridSize=NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    initParams.mode=NV_OF_MODE_OPTICALFLOW;
    initParams.inputBufferFormat=NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    initParams.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
    initParams.enableOutputCost = NV_OF_TRUE;
    CHECK_OF(nvOFAPI.nvOFInit(v->hOf, &initParams));

    create_nvof_buffer(v, width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, &v->inputFrame);
    create_nvof_buffer(v, width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, &v->referenceFrame);
    create_nvof_buffer(v, v->outW, v->outH, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, &v->flowBuf);
    create_nvof_buffer(v, v->outW, v->outH, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_UINT8, &v->costBuf);
    cuMemAllocHost((void**)&v->flowBufHost, 4*v->outW*v->outH);
    cuMemAllocHost((void**)&v->costBufHost, 1*v->outW*v->outH);
    v->count=0;
}

static int get_buf_stride(NvOFGPUBufferHandle buf)
{
    NV_OF_CUDA_BUFFER_STRIDE_INFO si;
    CHECK_OF(nvOFAPI.nvOFGPUBufferGetStrideInfo(buf, &si));
    return si.strideInfo[0].strideXInBytes;
}

nvof_results_t *nvof_execute(nvof_t *v, image_t *img)
{
    image_t *converted=image_convert(img, IMAGE_FORMAT_YUV420_DEVICE);
    assert(converted!=0);
    image_t *converted_scaled=image_scale(converted, v->width, v->height);
    assert(converted_scaled!=0);
    destroy_image(converted);

    nvof_set_size(v, v->width, v->height);

    NV_OF_EXECUTE_INPUT_PARAMS executeInParams = { 0 };
    NV_OF_EXECUTE_OUTPUT_PARAMS executeOutParams = { 0 };

    executeInParams.inputFrame=v->inputFrame;
    executeInParams.referenceFrame=v->referenceFrame;
    executeInParams.disableTemporalHints=NV_OF_TRUE;
    executeOutParams.outputBuffer = v->flowBuf;
    executeOutParams.outputCostBuffer = v->costBuf;

    CUDA_MEMCPY2D copyP;
    memset(&copyP, 0, sizeof(copyP));
    copyP.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyP.srcDevice = (CUdeviceptr)converted_scaled->device_mem;
    copyP.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyP.dstDevice = nvOFAPI.nvOFGPUBufferGetCUdeviceptr(v->inputFrame);
    copyP.srcPitch = converted_scaled->stride_y;
    copyP.dstPitch = get_buf_stride(v->inputFrame);
    copyP.WidthInBytes = converted_scaled->width;
    copyP.Height = converted_scaled->height;
    assert(CUDA_SUCCESS==cuMemcpy2DAsync(&copyP, converted_scaled->stream));
    if (v->count>0)
    {
        CHECK_OF(nvOFAPI.nvOFSetIOCudaStreams(v->hOf, converted_scaled->stream, converted_scaled->stream));
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
        assert(CUDA_SUCCESS==cuMemcpy2DAsync(&copyP1, converted_scaled->stream));

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
        assert(CUDA_SUCCESS==cuMemcpy2DAsync(&copyP2, converted_scaled->stream));
        
        cuStreamSynchronize(converted_scaled->stream);
    }
    else
    {
        cuStreamSynchronize(converted_scaled->stream);
    }
    
    v->count++;
    NvOFGPUBufferHandle temp=v->inputFrame;
    v->inputFrame=v->referenceFrame;
    v->referenceFrame=temp;

    destroy_image(converted_scaled);

    v->results.grid_w=v->outW;
    v->results.grid_h=v->outH;
    v->results.costs=v->costBufHost;
    v->results.flow=(flow_vector_t*)v->flowBufHost;
    return &v->results;
}

nvof_t *nvof_create(void *context, int width, int height) 
{
    check_cuda_inited();
    std::call_once(initFlag, init_nvof);

    nvof_t *n = (nvof_t *)malloc(sizeof(nvof_t));
    if (n==0) return 0;
    memset(n, 0, sizeof(nvof_t));

    CHECK_OF(nvOFAPI.nvCreateOpticalFlowCuda(get_CUcontext(), &n->hOf));

    nvof_set_size(n, width, height);

    return n;
}

void nvof_destroy(nvof_t *n) 
{
    if (n) 
    {
        destroy_nvof_buffer(n, &n->inputFrame);
        destroy_nvof_buffer(n, &n->referenceFrame);
        destroy_nvof_buffer(n, &n->flowBuf);
        destroy_nvof_buffer(n, &n->costBuf);
        if (n->flowBufHost) cuMemFreeHost(n->flowBufHost);
        if (n->costBufHost) cuMemFreeHost(n->costBufHost);
        CHECK_OF(nvOFAPI.nvOFDestroy(n->hOf));
        free(n);
    }
}

