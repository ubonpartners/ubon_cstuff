#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <mutex>
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

// Maximum number of CUDA events for the circular buffer.
// Use a power-of-two value to enable efficient indexing.
#define NUM_EVENTS 128

// Structure to hold context for CUDA stream dependency management.
typedef struct cuda_stuff_context {
    bool force_default_stream;
    bool force_sync;
    pthread_mutex_t lock;       // Mutex to protect concurrent access.
    uint32_t event_ct;          // Event counter for circular buffer indexing.
    uint32_t event_not_ready;   
    cudaEvent_t events[NUM_EVENTS]; // Pre-created CUDA events.
} cuda_stuff_context_t;

static cuda_stuff_context_t cs;
static std::once_flag initFlag;

// cuda_stream_add_dependency : allows you to fire-and-forget say that
// future work run on "stream_depends_on" can use anything previously
// issued on "stream"

// the intended model is that each object e.g. videosurface, has it's own
// cudaStream_t, that stream is used to generate that object's contents. You
// can use cuda_stream_add_dependency to add dependencies on the source 
// objects used

// for example suppose we have surfaceA and surfaceB
// surfaceA->stream and surfaceB->stream are the cudaStream_ts used to create
// the contents of A and B (we don't need to know how or what kernel)
// Supposed we want run a kernel to make surface C from a mix of A an B.
// 1) create empty surface C with C->stream
// 2) call cuda_stream_add_dependency(C->stream, A->stream)
//    AND call cuda_stream_add_dependency(C->stream, B->stream)
// 3) run kernel on stream C to generate the contents of C (using A/B data)
//
// The benefit of this scheme is you don't need to know in advance how what
// the dependencies will be

//--------------------------------------------------------------------
// Function: cuda_stream_add_dependency
//
// Purpose:
//   Establishes a dependency between two CUDA streams. The future work
//   on 'stream' will wait until all previously issued work on 
//   'stream_depends_on' (tracked by an event) is complete.
//
// Process:
//   1. Uses a circular buffer of CUDA events (with size NUM_EVENTS).
//   2. Synchronizes the reused event before recording a new dependency.
//   3. Records an event on 'stream_depends_on' and makes 'stream' wait
//      on that event.
//--------------------------------------------------------------------
void cuda_stream_add_dependency(cudaStream_t stream, cudaStream_t stream_depends_on)
{
    //cudaDeviceSynchronize();
    pthread_mutex_lock(&cs.lock);
    int index=cs.event_ct&(NUM_EVENTS-1);
    cs.event_ct++;
    //log_debug("Add cuda dependency for index %d\n",index);
    if (cudaEventQuery(cs.events[index])!=cudaSuccess)
    {
        CHECK_CUDART_CALL(cudaEventSynchronize(cs.events[index])); // should rarely happen
        cs.event_not_ready++;
        log_debug("cuda event was not ready %d/%d",cs.event_not_ready,cs.event_ct);
    }

    CHECK_CUDART_CALL(cudaEventRecord(cs.events[index], stream_depends_on));
    CHECK_CUDART_CALL(cudaStreamWaitEvent(stream, cs.events[index], 0));
    pthread_mutex_unlock(&cs.lock);
}

static void do_cuda_init()
{
    log_debug("Cuda init");

    // Set the CUDA device
    int device=0;
    cudaSetDevice(device);
    
    // Retrieve device properties to populate the NPP stream context.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Zero out context and set device attributes.
    memset(&nppStreamCtx, 0, sizeof(nppStreamCtx));
    nppStreamCtx.hStream = 0; // Using default stream; consider cudaStreamCreate() if needed.
    nppStreamCtx.nCudaDeviceId = device;
    nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMinor = prop.minor;

    // Initialize mutex.
    assert(0==pthread_mutex_init(&cs.lock, NULL));

    // Create a pool of CUDA events for dependency tracking.
    for(int i=0;i<NUM_EVENTS;i++)
        CHECK_CUDART_CALL(cudaEventCreateWithFlags(&cs.events[i], cudaEventDisableTiming));

    cs.force_sync=false;
    cs.force_default_stream=false;

    cuda_inited=true;
}

//--------------------------------------------------------------------
// Function: init_cuda_stuff
//
// Purpose:
//   Ensures that CUDA initialization and event creation happen only once.
//--------------------------------------------------------------------
void init_cuda_stuff()
{
    std::call_once(initFlag, do_cuda_init);
}

void cuda_set_sync_mode(bool force_sync, bool force_default_stream)
{
    init_cuda_stuff();
    cs.force_sync=force_sync;
    cs.force_default_stream=force_default_stream;
    log_debug("cuda_set_sync_mode force_sync:%d force_default_stream:%d", cs.force_sync, cs.force_default_stream);
    cudaDeviceSynchronize();
}

void check_cuda_inited()
{
    assert(cuda_inited);
}

CUcontext get_CUcontext()
{
    assert(cuda_inited);
    CUcontext cuContext;
    CUresult res = cuCtxGetCurrent(&cuContext);
    assert(res== CUDA_SUCCESS);
    return cuContext;
}

cudaStream_t create_cuda_stream()
{
    cudaStream_t ret = 0;
    if (cs.force_sync) cudaDeviceSynchronize();
    if (cs.force_default_stream) return ret;
    CHECK_CUDART_CALL(cudaStreamCreate(&ret));
    return ret;
}

void destroy_cuda_stream(cudaStream_t stream)
{
    if (stream==0) return;
    CHECK_CUDART_CALL(cudaStreamDestroy(stream));
}

