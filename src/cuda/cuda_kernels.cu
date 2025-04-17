#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "cuda.h"

extern "C" {

static __device__ inline float clamp(float x) {
    return x < 0 ? 0 : (x > 1 ? 1 : x);
}

// CUDA Kernel to convert YUV (8-bit unsigned int) to RGB (float16)
static __global__ void yuvToRgbKernel_fp16(const unsigned char* y_plane, const unsigned char* u_plane, const unsigned char* v_plane,
                               __half* r_plane, __half* g_plane, __half* b_plane,
                               int width, int height, int y_stride, int uv_stride) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Get 8-bit YUV values and normalize to the range [0, 1]
        float y_value = y_plane[y * y_stride + x];
        float u_value = u_plane[(y / 2) * uv_stride + (x / 2)];
        float v_value = v_plane[(y / 2) * uv_stride + (x / 2)];

        // Convert YUV to RGB
        float c = (y_value - 16.0f) / 255.0f;
        float d = (u_value - 128.0f) / 255.0f;
        float e = (v_value - 128.0f) / 255.0f;

        float r = clamp(1.164f * c + 1.596f * e);
        float g = clamp(1.164f * c - 0.392f * d - 0.813f * e);
        float b = clamp(1.164f * c + 2.017f * d);

        // Write RGB values to the output as float16
        r_plane[y * width + x] = __float2half(r);
        g_plane[y * width + x] = __float2half(g);
        b_plane[y * width + x] = __float2half(b);
    }
}

static __global__ void yuvToRgbKernel_fp32(const unsigned char* y_plane, const unsigned char* u_plane, const unsigned char* v_plane,
                               float* r_plane, float* g_plane, float* b_plane,
                               int width, int height, int y_stride, int uv_stride) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Get 8-bit YUV values and normalize to the range [0, 1]
        float y_value = y_plane[y * y_stride + x];
        float u_value = u_plane[(y / 2) * uv_stride + (x / 2)];
        float v_value = v_plane[(y / 2) * uv_stride + (x / 2)];

        // Convert YUV to RGB
        float c = (y_value - 16.0f) / 255.0f;
        float d = (u_value - 128.0f) / 255.0f;
        float e = (v_value - 128.0f) / 255.0f;

        float r = clamp(1.164f * c + 1.596f * e);
        float g = clamp(1.164f * c - 0.392f * d - 0.813f * e);
        float b = clamp(1.164f * c + 2.017f * d);

        r_plane[y * width + x] = r;
        g_plane[y * width + x] = g;
        b_plane[y * width + x] = b;
    }
}

// Function to launch the CUDA kernel
void cuda_convertYUVtoRGB_fp16(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
                     int y_stride, int uv_stride,
                     uint8_t *dest,
                     int width, int height, CUstream stream) 
{
    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    __half *d_r_plane=(__half *)dest;
    __half *d_g_plane=d_r_plane+width*height;
    __half *d_b_plane=d_g_plane+width*height;

    // Launch the kernel
    yuvToRgbKernel_fp16<<<grid, block, 0, stream>>>(d_y_plane, d_u_plane, d_v_plane,
                                    (__half*)d_r_plane, (__half*)d_g_plane, (__half*)d_b_plane,
                                    width, height, y_stride, uv_stride);

}

void cuda_convertYUVtoRGB_fp32(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
                     int y_stride, int uv_stride,
                     uint8_t *dest,
                     int width, int height, CUstream stream) 
{
    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    float *d_r_plane=(float *)dest;
    float *d_g_plane=d_r_plane+width*height;
    float *d_b_plane=d_g_plane+width*height;

    // Launch the kernel
    yuvToRgbKernel_fp32<<<grid, block, 0, stream>>>(d_y_plane, d_u_plane, d_v_plane,
                                    d_r_plane, d_g_plane, d_b_plane,
                                    width, height, y_stride, uv_stride);

}

static __global__ void half_to_float_kernel(const __half* d_input, float* d_output, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = __half2float(d_input[idx]);
    }
}

void cuda_half_to_float(void* d_input, void* h_output, int size, CUstream stream) 
{
    // Launch kernel with appropriate configuration
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    half_to_float_kernel<<<gridSize, blockSize, 0, stream>>>((const __half*)d_input, (float *)h_output, size);
}

static __global__ void fp16_planar_to_RGB24_kernel(const __half *src, unsigned char *dest, int dest_stride, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        float r=__half2float(src[x+y*width+width*height*0]);
        float g=__half2float(src[x+y*width+width*height*1]);
        float b=__half2float(src[x+y*width+width*height*2]);

        dest[x*3+dest_stride*y+0]=(unsigned char)(r*255);
        dest[x*3+dest_stride*y+1]=(unsigned char)(g*255);
        dest[x*3+dest_stride*y+2]=(unsigned char)(b*255);
    }
}

void cuda_convert_fp16_planar_to_RGB24(void *src, void *dest, int dest_stride, int width, int height, CUstream stream) 
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    fp16_planar_to_RGB24_kernel<<<grid, block, 0, stream>>>((const __half*)src, (unsigned char *)dest, dest_stride, width, height);
}

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
    CUstream stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

    downsample_2x2_kernel<<<grid, block, 0, stream>>>(
        d_src, src_stride,
        d_dst, dst_stride,
        dst_width, dst_height
    );
}

static __global__ void interleave_uv_kernel(
    const uint8_t* __restrict__ u,
    const uint8_t* __restrict__ v,
    int src_stride_uv,
    uint8_t* __restrict__ dst,
    int dest_stride_uv,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // pixel index in row
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index

    if (x < width && y < height) {
        const uint8_t u_val = u[y * src_stride_uv + x];
        const uint8_t v_val = v[y * src_stride_uv + x];

        int dst_idx = y * dest_stride_uv + 2 * x;
        dst[dst_idx]     = u_val;
        dst[dst_idx + 1] = v_val;
    }
}

void cuda_interleave_uv(
    const uint8_t* d_u, const uint8_t* d_v,
    int src_stride_uv,
    uint8_t* d_dst,
    int dest_stride_uv,
    int width,
    int height,
    CUstream stream)
{
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    interleave_uv_kernel<<<grid, block, 0, stream>>>(
        d_u, d_v,
        src_stride_uv,
        d_dst,
        dest_stride_uv,
        width,
        height
    );
}

static __global__ void fnv1a_chunk_hash_kernel(
    const uint8_t* data,
    size_t size,
    size_t chunk_size,
    uint32_t* partial_hashes
) {
    size_t block_id = blockIdx.x;
    size_t offset = block_id * chunk_size;

    if (offset >= size)
        return;

    size_t end = min(offset + chunk_size, size);
    uint32_t hash = 2166136261u;

    // Let one thread do the hash (or loop over threads if needed)
    if (threadIdx.x == 0) {
        int num=(end-offset)/4;
        uint32_t *p=(uint32_t *)(data+offset);
        for (int i=0;i<num;i++)
        {
            hash ^= p[i];
            hash *= 16777619u;
        }
        partial_hashes[block_id] = hash;
    }
}

uint32_t hash_gpu(void* d_data, int size_bytes, CUstream stream)
{
    size_t chunk_size=4096;
    size_t num_chunks = (size_bytes + chunk_size - 1) / chunk_size;

    uint32_t* d_partials;
    uint32_t h_partials[num_chunks];

    cudaMallocAsync(&d_partials, num_chunks * sizeof(uint32_t), stream);

    fnv1a_chunk_hash_kernel<<<num_chunks, 1, 0, stream>>>(
        static_cast<const uint8_t*>(d_data),
        size_bytes,
        chunk_size,
        d_partials
    );

    cudaMemcpyAsync(h_partials, d_partials, num_chunks * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_partials, stream);

    // Combine partial hashes on host
    uint32_t final_hash = 2166136261u;
    for (int i=0;i<num_chunks;i++) 
    {
        final_hash ^= h_partials[i];
        final_hash *= 16777619u;
    }
    
    return final_hash;
}

}



