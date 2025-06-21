#ifndef __TRT_STUFF
#define __TRT_STUFF

#include "log.h"
#include "cuda_stuff.h"
#include "memory_stuff.h"

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept
    {
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
            log_error("[TRT] %s\n",msg);
        if (severity == Severity::kWARNING)
            log_warn("[TRT] %s\n",msg);
    }
};

class TRTAllocator : public nvinfer1::IGpuAllocator // todo: use IGpuAsyncAllocator!
{
public:
    TRTAllocator(allocation_tracker_t *t)
    {
        allocation_tracker=t;
    }

    ~TRTAllocator() override = default;

    allocation_tracker_t * allocation_tracker;

    void* allocate(uint64_t size, uint64_t alignment, nvinfer1::AllocatorFlags flags) noexcept override
    {
        void *p=cuda_malloc(size);
        track_alloc_table(allocation_tracker, size, p);
        return p;
    }

    bool deallocate(void* memory) noexcept override
    {
        cuda_free(memory);
        track_free_table(allocation_tracker, memory);
        return true;
    }

    void* reallocate(void *ptr, uint64_t old_size, uint64_t new_size)  noexcept override
    {
        return(0);
    }
};

extern Logger trt_Logger;
extern TRTAllocator trt_allocator;

void trt_init();

#endif
