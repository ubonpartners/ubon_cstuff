#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <nvml.h>
#include <unistd.h>
#include <mutex>
#include <cuda.h>
#include <cassert>
#include "cuda_stuff.h"
#include "pthread.h"
#include "misc.h"
#include "memory_stuff.h"

#if (UBONCSTUFF_PLATFORM == 0) // Desktop Nvidia GPU
#define USE_NVML    1
#else
#undef USE_NVML
#endif

static bool cuda_inited=false;
static allocation_tracker_t cuda_alloc_tracker;
static allocation_tracker_t cuda_alloc_async_tracker;
static allocation_tracker_t cuda_alloc_host_tracker;
static pthread_t cuda_thread;
NppStreamContext nppStreamCtx;
#ifdef USE_NVML
static nvmlDevice_t nvml_device;
#endif

NppStreamContext get_nppStreamCtx()
{
    return nppStreamCtx;
}

// Maximum number of CUDA events for the circular buffer.
// Use a power-of-two value to enable efficient indexing.
#define NUM_EVENTS  128
#define NUM_STREAMS 128

// Structure to hold context for CUDA stream dependency management.
typedef struct cuda_stuff_context {
    bool force_default_stream;
    bool force_sync;
    cudaStream_t stream;
    CUmemoryPool defaultPool;
    CUmemoryPool appPool;      // cuda memory pool we use for our allocations
    pthread_mutex_t lock;       // Mutex to protect concurrent access.
    pthread_mutex_t stream_lock;       // Mutex to protect concurrent access.
    uint32_t event_ct;          // Event counter for circular buffer indexing.
    uint32_t event_not_ready;
    uint32_t stream_ct;
    uint32_t stream_not_ready;
    uint64_t process_gpu_memory_hwm;
    uint64_t process_gpu_memory_baseline;
    cudaEvent_t events[NUM_EVENTS]; // Pre-created CUDA events.
    cudaStream_t streams[NUM_STREAMS];
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
        cs.event_not_ready++;
        log_warn("cuda event was not ready %d/%d",cs.event_not_ready,cs.event_ct);
        CHECK_CUDART_CALL(cudaEventSynchronize(cs.events[index])); // should rarely happen
    }

    CHECK_CUDART_CALL(cudaEventRecord(cs.events[index], stream_depends_on));
    CHECK_CUDART_CALL(cudaStreamWaitEvent(stream, cs.events[index], 0));
    pthread_mutex_unlock(&cs.lock);
}

static uint64_t get_process_GPU_memory()
{
    #ifdef USE_NVML
    nvmlReturn_t result;
    unsigned int infoCount = 100;
    nvmlProcessInfo_t infos[100];

    result = nvmlDeviceGetComputeRunningProcesses(nvml_device, &infoCount, infos);
    if (result != NVML_SUCCESS && result != NVML_ERROR_INSUFFICIENT_SIZE) {
        printf("Failed to get compute processes: %s\n", nvmlErrorString(result));
    }

    unsigned int my_pid = getpid();

    // Search for this process
    for (unsigned int i = 0; i < infoCount; ++i) {
        if (infos[i].pid == my_pid) {
            if ((unsigned long long)infos[i].usedGpuMemory)
            return (unsigned long long)infos[i].usedGpuMemory;
        }
    }
    #endif
    return 0;
}
static void *cuda_thread_fn(void *arg)
{
    while(1)
    {
        usleep(100*1000);

        uint64_t process_gpu_memory=get_process_GPU_memory();
        if (process_gpu_memory>cs.process_gpu_memory_hwm)
        {
            cs.process_gpu_memory_hwm=process_gpu_memory;
        }
    }
    return 0;
}

static void do_cuda_init()
{
    log_debug("Cuda init");

    // --------------------------------------------------
    // Start NVML
    // --------------------------------------------------
    #ifdef USE_NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        log_fatal("Failed to initialize NVML: %s", nvmlErrorString(result));
    }
    result = nvmlDeviceGetHandleByIndex(0, &nvml_device);
    if (result != NVML_SUCCESS) {
        log_fatal("Failed to get NVML handle for device 0: %s", nvmlErrorString(result));
    }
    #endif
    log_trace("Initial process GPU memory usage %5.1fMB",get_process_GPU_memory()/(1024.0*1024.0));

    // --------------------------------------------------
    // Register allocation trackers
    // --------------------------------------------------

    allocation_tracker_register(&cuda_alloc_tracker, "cuda alloc", true);
    allocation_tracker_register(&cuda_alloc_async_tracker, "cuda async alloc", true);
    allocation_tracker_register(&cuda_alloc_host_tracker, "cuda alloc host", true);

    // --------------------------------------------------
    // Initialize Driver API context (Pool A is the default)
    // --------------------------------------------------

    CHECK_CUDA_CALL(cuInit(0));
    CUdevice cuDev;
    CHECK_CUDA_CALL(cuDeviceGet(&cuDev, 0));
    CUcontext cuCtx;
    #if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
    CHECK_CUDA_CALL(cuCtxCreate(&cuCtx, 0, 0, cuDev));
    #else
    CHECK_CUDA_CALL(cuCtxCreate(&cuCtx, 0, cuDev));
    #endif

    // --------------------------------------------------
    // Bind device for Runtime API
    // --------------------------------------------------
    int device = 0;
    CHECK_CUDART_CALL(cudaSetDevice(device));

    // --------------------------------------------------
    // Get default pool and
    // Create cuda memory pool for our cudaMalloc/cudaMallocAsync
    // --------------------------------------------------
    log_trace("GPU memory before pool creation: %5.1fMB",get_process_GPU_memory()/(1024.0*1024.0));
    CHECK_CUDA_CALL(cuDeviceGetDefaultMemPool(&cs.defaultPool, cuDev));

     {
      CUmemPoolProps props = {};
      props.allocType     = CU_MEM_ALLOCATION_TYPE_PINNED;
      props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      props.location.id   = device;
      props.handleTypes   = CU_MEM_HANDLE_TYPE_NONE;

      CHECK_CUDA_CALL(cuMemPoolCreate(&cs.appPool, &props));
      // Make it the default for cudaMallocAsync/cudaFreeAsync
      //CHECK_CUDART_CALL(cudaDeviceSetMemPool(device, (cudaMemPool_t)&cs.appPool));
    }

    // --------------------------------------------------
    // Build your NPP stream context
    // --------------------------------------------------

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    log_debug("CUDA bound to GPU %d: %s (CC %d.%d)", device, prop.name, prop.major, prop.minor);


    memset(&nppStreamCtx, 0, sizeof(nppStreamCtx));
    nppStreamCtx.hStream = 0; // Using default stream; consider cudaStreamCreate() if needed.
    nppStreamCtx.nCudaDeviceId = device;
    nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    nppStreamCtx.nCudaDevAttrComputeCapabilityMinor = prop.minor;



    // --------------------------------------------------
    // Initialize your event pool and mutex (unchanged)
    // --------------------------------------------------

    assert(0==pthread_mutex_init(&cs.lock, NULL));
    assert(0==pthread_mutex_init(&cs.stream_lock, NULL));

    // Create a pool of CUDA events for dependency tracking.
    for(int i=0;i<NUM_EVENTS;i++)
        CHECK_CUDART_CALL(cudaEventCreateWithFlags(&cs.events[i], cudaEventDisableTiming));

    cs.force_sync=false;
    cs.force_default_stream=false;

    CHECK_CUDART_CALL(cudaStreamCreate(&cs.stream));
    for(int i=0;i<NUM_EVENTS;i++)
        CHECK_CUDART_CALL(cudaStreamCreate(&cs.streams[i]));

    log_debug("GPU memory usage after event+stream creation: %5.1fMB",get_process_GPU_memory()/(1024.0*1024.0));
    cs.process_gpu_memory_baseline=get_process_GPU_memory();

    cuda_inited=true;
    pthread_create(&cuda_thread, NULL, cuda_thread_fn, 0);
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

cudaStream_t create_cuda_stream_pool()
{
    uint32_t prev=__sync_fetch_and_add(&cs.stream_ct, 1);
    int index=prev&(NUM_STREAMS-1);
    //pthread_mutex_unlock(&cs.lock);
    return cs.streams[index];
}

void destroy_cuda_stream_pool(cudaStream_t stream)
{
    // don't need to do anything!
    // it's perfectly ok if the pool wrapps and a stream gets used by two things
}

void cuda_thread_init()
{
    // Ensure global init
    init_cuda_stuff();

    // Check if a CUDA context is already current
    CUcontext ctx = nullptr;
    CUresult res = cuCtxGetCurrent(&ctx);
    if (res == CUDA_SUCCESS && ctx != nullptr) {
        // Context is already active on this thread — nothing more needed
        return;
    }

    // Bind device and force context creation
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        log_fatal("cuda_thread_init: cudaSetDevice(%d) failed: %s",device, cudaGetErrorString(err));
        abort();
    }

    // Force creation of primary context on this thread
    err = cudaFree(0);
    if (err != cudaSuccess) {
        log_fatal("cuda_thread_init: cudaFree(0) failed: %s",cudaGetErrorString(err));
        abort();
    }

    // Optional: sanity check
    res = cuCtxGetCurrent(&ctx);
    if (res != CUDA_SUCCESS || ctx == nullptr) {
        log_fatal("cuda_thread_init: cuCtxGetCurrent failed: %d\n", res);
        abort();
    }


    log_trace("CUDA context initialized on thread %lu (context: %p)",(unsigned long)pthread_self(), (void*)ctx);
}

void *cuda_malloc(size_t size)
{
    void *ptr=0;
    assert(size>0);
    CHECK_CUDART_CALL(cudaMallocFromPoolAsync(&ptr, size, cs.appPool, cs.stream));
    CHECK_CUDART_CALL(cudaStreamSynchronize(cs.stream));
    track_alloc_table(&cuda_alloc_tracker, size, ptr);
    return ptr;
}

void *cuda_malloc_async(size_t size, cudaStream_t stream)
{
    void *ptr=0;
    CHECK_CUDART_CALL(cudaMallocFromPoolAsync(&ptr, size, cs.appPool, stream));
    track_alloc_table(&cuda_alloc_async_tracker, size, ptr);
    return ptr;
}

void *cuda_malloc_host(size_t size)
{
    void *ptr=0;
    CHECK_CUDART_CALL(cudaMallocHost(&ptr, size));
    track_alloc_table(&cuda_alloc_host_tracker, size, ptr);
    return ptr;
}

void cuda_malloc_check(void *ptr)
{
    if (ptr==0) return;
    track_check(&cuda_alloc_tracker, ptr);
}

void cuda_malloc_async_check(void *ptr)
{
    if (ptr==0) return;
    track_check(&cuda_alloc_async_tracker, ptr);
}

void cuda_free(void *ptr)
{
    if (ptr==0) return;
    track_free_table(&cuda_alloc_tracker, ptr);
    CHECK_CUDART_CALL(cudaFreeAsync(ptr, cs.stream));
    CHECK_CUDART_CALL(cudaStreamSynchronize(cs.stream));
}

void cuda_free_async(void *ptr, cudaStream_t stream)
{
    if (ptr==0) return;
    track_free_table(&cuda_alloc_async_tracker, ptr);
    CHECK_CUDART_CALL(cudaFreeAsync(ptr, stream));
}

void cuda_free_host(void *ptr)
{
    if (ptr==0) return;
    track_free_table(&cuda_alloc_host_tracker, ptr);
    CHECK_CUDART_CALL(cudaFreeHost(ptr));
}

void cuda_flush()
{
    CHECK_CUDA_CALL(cuMemPoolTrimTo(cs.appPool, 0));
}

double get_cuda_mem(bool default_pool, bool hwm, bool reset)
{
    size_t used=0;
    uint64_t zero=0;

    CHECK_CUDA_CALL(cuMemPoolGetAttribute(default_pool ? cs.defaultPool : cs.appPool,
        hwm ? CU_MEMPOOL_ATTR_USED_MEM_HIGH : CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &used));

    if (reset)
    {
        CHECK_CUDA_CALL(cuMemPoolSetAttribute(default_pool ? cs.defaultPool : cs.appPool, CU_MEMPOOL_ATTR_USED_MEM_HIGH, &zero));

    }
    return (double)used;
}

double get_process_gpu_mem(bool hwm, bool reset)
{
    if (hwm)
    {
        double ret=(double)cs.process_gpu_memory_hwm;
        if (reset)
        {
            cuda_flush();
            cs.process_gpu_memory_hwm=0;
        }
        //ret-=(double)cs.process_gpu_memory_baseline;
        return ret;
    }
    return (double)get_process_GPU_memory()/*-(double)cs.process_gpu_memory_baseline*/;
}
