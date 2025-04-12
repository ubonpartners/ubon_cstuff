#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cassert>
#include "cuda_stuff.h"
#include "pthread.h"

static bool cuda_inited=false;

NppStreamContext nppStreamCtx;

NppStreamContext get_nppStreamCtx()
{
    return nppStreamCtx;
}

#define NUM_EVENTS  128

typedef struct cuda_stuff_context
{
    pthread_mutex_t lock;
    uint32_t event_ct;
    CUevent events[NUM_EVENTS];
} cuda_stuff_context_t;

static cuda_stuff_context_t cs;

static void cuda_event_init()
{
    assert(0==pthread_mutex_init(&cs.lock, NULL));
    for(int i=0;i<NUM_EVENTS;i++)
        CHECK_CUDA_CALL(cuEventCreate(&cs.events[i], CU_EVENT_DEFAULT));
}

void cuda_stream_add_dependency(CUstream stream, CUstream stream_depends_on)
{
    pthread_mutex_lock(&cs.lock);
    int index=cs.event_ct&(NUM_EVENTS-1);
    cs.event_ct++;
    CHECK_CUDA_CALL(cuEventSynchronize(cs.events[index]));
    cudaEventRecord(cs.events[index], stream_depends_on);
    cudaStreamWaitEvent(stream, cs.events[index], 0);
    pthread_mutex_unlock(&cs.lock);
}

void init_cuda_stuff()
{
    CUcontext cuContext;
    CHECK_CUDA_CALL(cuInit(0));
    CHECK_CUDA_CALL(cuCtxCreate(&cuContext, 0, 0));
    
    // Set the CUDA device
    int device = 0;
    cudaSetDevice(device);
    
    // Initialize stream context for the chosen device
    nppStreamCtx.hStream = 0; // Default stream or you can create one with cudaStreamCreate()
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    

    memset(&nppStreamCtx, 0, sizeof(nppStreamCtx));
    nppStreamCtx.nCudaDeviceId = device;
    nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    //nppStreamCtx.nCudaDevAttrWarpSize = prop.warpSize;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMinor = prop.minor;

    cuda_event_init();
    cuda_inited=true;
}

void check_cuda_inited()
{
    assert(cuda_inited);
}