#ifndef __CUDA_STUFF_H
#define __CUDA_STUFF_H

#include <npp.h>
#include <cuda.h>

// Error checking macro
#define CHECK_CUDA_CALL(call) do {      \
    CUresult err = call;                \
    if (err != CUDA_SUCCESS) {          \
        const char *errStr;             \
        cuGetErrorName(err, &errStr);   \
        fprintf(stderr, "CUDA Error: %s at line %d\n", errStr, __LINE__); \
        exit(EXIT_FAILURE);             \
        }                               \
} while (0)

void check_cuda_inited();
void init_cuda_stuff();
NppStreamContext get_nppStreamCtx();
void cuda_stream_add_dependency(CUstream stream, CUstream stream_depends_on);
CUcontext get_CUcontext();

#endif