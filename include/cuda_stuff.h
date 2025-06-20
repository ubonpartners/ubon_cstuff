#ifndef __CUDA_STUFF_H
#define __CUDA_STUFF_H

#include <npp.h>
#include <cuda.h>
#include "log.h"

#define CHECK_CUDA_CALL(call) \
    do { \
        CUresult _status = (call); \
        if (_status != CUDA_SUCCESS) { \
            const char *namestr,*errstr; \
            cuGetErrorName(_status, &namestr); \
            cuGetErrorString(_status, &errstr); \
            log_fatal("Cuda error code %d %s:%s",_status,namestr,errstr); \
            exit(1); \
        }  \
    } while(0)

#define CHECK_CUDART_CALL(call) \
    do { \
        cudaError_t _status = (call); \
        if (_status != cudaSuccess) { \
            log_fatal("Cuda error code %d %s",_status,cudaGetErrorString(_status)); \
            exit(1); \
        }  \
    } while(0)

void check_cuda_inited();
void init_cuda_stuff();
NppStreamContext get_nppStreamCtx();
void cuda_stream_add_dependency(cudaStream_t stream, cudaStream_t stream_depends_on);
CUcontext get_CUcontext();
cudaStream_t create_cuda_stream();
void destroy_cuda_stream(cudaStream_t s);
void cuda_set_sync_mode(bool force_sync, bool force_default_stream);
void cuda_thread_init();

void *cuda_malloc(size_t size);
void *cuda_malloc_host(size_t size);
void cuda_free(void *p);
void cuda_free_host(void *p);

#endif