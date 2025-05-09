#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include "cuda.h"

extern "C" {

static __global__ void downsample_2x2_kernel(const uint8_t* src, int src_stride,
    uint8_t* dst, int dst_stride,
    int dst_width, int dst_height)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x; // column in dst
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y; // row in dst

    if (dst_x < dst_width && dst_y < dst_height ) {
        // Source top-left of the 2x2 block
        int src_x = dst_x * 2;
        int src_y = dst_y * 2;

        int idx0 = src_y * src_stride + src_x;
        int idx1 = idx0 + 1;
        int idx2 = idx0 + src_stride;
        int idx3 = idx2 + 1;

        // Average the 2x2 block and store to destination
        uint8_t a = src[idx0];
        uint8_t b = src[idx1];
        uint8_t c = src[idx2];
        uint8_t d = src[idx3];

        dst[dst_y * dst_stride + dst_x] = (a + b + c + d + 2) / 4; // +2 for rounding
    }
}

void cuda_downsample_2x2(const uint8_t* d_src, int src_stride,
        uint8_t* d_dst, int dst_stride,
        int dst_width, int dst_height,
        cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

    downsample_2x2_kernel<<<grid, block, 0, stream>>>(
        d_src, src_stride,
        d_dst, dst_stride,
        dst_width, dst_height
    );
}

} // extern "C"