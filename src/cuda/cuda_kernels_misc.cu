#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "cuda.h"

extern "C" {

#define FNV_OFFSET 2166136261u
#define FNV_PRIME 16777619u

// Kernel: Each thread computes a hash for a single row
static __global__ void row_hash_kernel(const uint8_t* data, int w, int h, int stride, uint32_t* row_hashes)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < h)
    {
        const uint8_t* row_ptr = data + row * stride;

        uint32_t hash = FNV_OFFSET;
        for (int i = 0; i < w; ++i)
        {
            hash ^= row_ptr[i];
            hash *= FNV_PRIME;
        }

        row_hashes[row] = hash;
    }
}

void cuda_hash_2d(const uint8_t* d_data, int w, int h, int stride, uint32_t *dest, cudaStream_t stream)
{
    uint32_t *dest_device=0;
    cudaMallocAsync(&dest_device, h * sizeof(uint32_t), stream);
    int threads = 256;
    int blocks = (h + threads - 1) / threads;
    row_hash_kernel<<<blocks, threads, 0, stream>>>(d_data, w, h, stride, dest_device);
    cudaMemcpyAsync(dest, dest_device, h * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(dest, stream);
}

// Kernel to compute MAD per 4x4 block
static __global__ void kernel_block_mad_4x4(
    const uint8_t *img1, int stride1,
    const uint8_t *img2, int stride2,
    uint8_t *out, int out_stride,
    int out_width, int out_height)
{
    int bx = blockIdx.x * blockDim.x + threadIdx.x;  // output x (block index)
    int by = blockIdx.y * blockDim.y + threadIdx.y;  // output y (block index)

    if (bx >= out_width || by >= out_height) return;

    int x = bx * 4;
    int y = by * 4;

    const uint32_t *p1 = (const uint32_t *)(img1 + y*stride1 + x);
    const uint32_t *p2 = (const uint32_t *)(img2 + y*stride2 + x);
    stride1>>=2;
    stride2>>=2;
    uint32_t diff0=__vabsdiffu4(p1[0*stride1], p2[0*stride2]);
    uint32_t diff1=__vabsdiffu4(p1[1*stride1], p2[1*stride2]);
    uint32_t diff2=__vabsdiffu4(p1[2*stride1], p2[2*stride2]);
    uint32_t diff3=__vabsdiffu4(p1[3*stride1], p2[3*stride2]);
    uint32_t diff01=__vavgu4(diff0, diff1);
    uint32_t diff23=__vavgu4(diff2, diff3);
    uint32_t diff=__vavgu4(diff01, diff23);
    uint32_t sum = (diff & 0xff) +
            ((diff >> 8) & 0xff) +
            ((diff >> 16) & 0xff) +
            ((diff >> 24) & 0xff);
    out[by * out_stride + bx] = ((sum+2)>>2);
}

void compute_4x4_mad_mask(uint8_t *a, int stride_a, uint8_t *b, int stride_b,
                          uint8_t *out, int stride_out, int width, int height, cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    kernel_block_mad_4x4<<<gridDim, blockDim, 0, stream>>>(
        a, stride_a,
        b, stride_b,
        out, stride_out,
        width, height);
}

// For float (fp32)
static __global__ void set_rgb_region_float(
    float* base_ptr, int w, int h,
    int dst_w, int dst_plane_size_elements, float val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
        return;

    int offset = y * dst_w + x;

    base_ptr[offset] = val;
    base_ptr[offset + dst_plane_size_elements] = val;
    base_ptr[offset + 2 * dst_plane_size_elements] = val;
}

// For half (fp16)
static __global__ void set_rgb_region_half(
    __half* base_ptr, int w, int h,
    int dst_w, int dst_plane_size_elements, __half val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
        return;

    int offset = y * dst_w + x;

    base_ptr[offset] = val;
    base_ptr[offset + dst_plane_size_elements] = val;
    base_ptr[offset + 2 * dst_plane_size_elements] = val;
}

void cuda_fp_set(
    void* rgb_plane_ptr, int w, int h,
    int dst_w, int dst_plane_size_elements,
    cudaStream_t stream, bool is_fp16)
{
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    if (is_fp16) {
        __half val = __float2half(0.5f);
        set_rgb_region_half<<<grid, block, 0, stream>>>(
            static_cast<__half*>(rgb_plane_ptr),
            w, h, dst_w, dst_plane_size_elements, val);
    } else {
        float val = 0.5f;
        set_rgb_region_float<<<grid, block, 0, stream>>>(
            static_cast<float*>(rgb_plane_ptr),
            w, h, dst_w, dst_plane_size_elements, val);
    }
}

} // extern "C"
