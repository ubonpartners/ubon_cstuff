#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "cuda.h"

extern "C" {

static __device__ inline float clamp(float x) {
    return x < 0 ? 0 : (x > 1 ? 1 : x);
}

// CUDA Kernel to convert YUV (8-bit unsigned int, BT.709) to RGB (float16)
static __global__ void yuvToRgbKernel_fp16(const unsigned char* y_plane, const unsigned char* u_plane, const unsigned char* v_plane,
                               __half* r_plane, __half* g_plane, __half* b_plane,
                               int width, int height, int y_stride, int uv_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Load YUV values from planes
        int y_index = y * y_stride + x;
        int uv_index = (y / 2) * uv_stride + (x / 2);

        float y_value = static_cast<float>(y_plane[y_index]);
        float u_value = static_cast<float>(u_plane[uv_index]);
        float v_value = static_cast<float>(v_plane[uv_index]);

        // BT.709 conversion (limited range)
        float c = y_value - 16.0f;
        float d = u_value - 128.0f;
        float e = v_value - 128.0f;

        float r = (1.164f * c + 1.793f * e) / 255.0f;
        float g = (1.164f * c - 0.213f * d - 0.533f * e) / 255.0f;
        float b = (1.164f * c + 2.112f * d) / 255.0f;

        r_plane[y * width + x] = __float2half(clamp(r));
        g_plane[y * width + x] = __float2half(clamp(g));
        b_plane[y * width + x] = __float2half(clamp(b));
    }
}

// CUDA Kernel to convert YUV (8-bit unsigned int, BT.709) to RGB (float32)
static __global__ void yuvToRgbKernel_fp32(const unsigned char* y_plane, const unsigned char* u_plane, const unsigned char* v_plane,
                               float* r_plane, float* g_plane, float* b_plane,
                               int width, int height, int y_stride, int uv_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Load YUV values from planes
        int y_index = y * y_stride + x;
        int uv_index = (y / 2) * uv_stride + (x / 2);

        float y_value = static_cast<float>(y_plane[y_index]);
        float u_value = static_cast<float>(u_plane[uv_index]);
        float v_value = static_cast<float>(v_plane[uv_index]);

        // BT.709 conversion (limited range)
        float c = y_value - 16.0f;
        float d = u_value - 128.0f;
        float e = v_value - 128.0f;

        float r = (1.164f * c + 1.793f * e) / 255.0f;
        float g = (1.164f * c - 0.213f * d - 0.533f * e) / 255.0f;
        float b = (1.164f * c + 2.112f * d) / 255.0f;

        r_plane[y * width + x] = clamp(r);
        g_plane[y * width + x] = clamp(g);
        b_plane[y * width + x] = clamp(b);
    }
}

// Function to launch the CUDA kernel
void cuda_convertYUVtoRGB_fp16(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
                     int y_stride, int uv_stride,
                     uint8_t *dest,
                     int width, int height, cudaStream_t stream)
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
                     int width, int height, cudaStream_t stream)
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

static __device__ uint8_t clamp_u8(float val) {
    return static_cast<uint8_t>(fminf(fmaxf(val, 0.0f), 255.0f));
}

static __global__ void yuv420_to_rgb24_kernel(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ u_plane,
    const uint8_t* __restrict__ v_plane,
    uint8_t* __restrict__ rgb_out,
    int width,
    int height,
    int y_stride,
    int uv_stride,
    int rgb_stride)  // number of bytes per row in output
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Get YUV values
    int y_index = y * y_stride + x;
    int uv_index = (y / 2) * uv_stride + (x / 2);

    float yf = static_cast<float>(y_plane[y_index]);
    float uf = static_cast<float>(u_plane[uv_index]) - 128.0f;
    float vf = static_cast<float>(v_plane[uv_index]) - 128.0f;

    // BT.709 limited-range conversion
    float c = yf - 16.0f;

    float r = 1.164f * c + 1.793f * vf;
    float g = 1.164f * c - 0.213f * uf - 0.533f * vf;
    float b = 1.164f * c + 2.112f * uf;

    uint8_t R = clamp_u8(r);
    uint8_t G = clamp_u8(g);
    uint8_t B = clamp_u8(b);

    // Write packed RGB output
    int rgb_index = y * rgb_stride + x * 3;
    rgb_out[rgb_index + 0] = R;
    rgb_out[rgb_index + 1] = G;
    rgb_out[rgb_index + 2] = B;
}

void cuda_convertYUVtoRGB24(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
    int y_stride, int uv_stride,
    uint8_t *dest, int dest_stride,
    int width, int height, cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);

    yuv420_to_rgb24_kernel<<<gridDim, blockDim>>>(
        d_y_plane, d_u_plane, d_v_plane, dest,
        width, height,
        y_stride, uv_stride, dest_stride);
}

static __global__ void half_to_float_kernel(const __half* d_input, float* d_output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = __half2float(d_input[idx]);
    }
}

void cuda_half_to_float(void* d_input, void* h_output, int size, cudaStream_t stream)
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

void cuda_convert_fp16_planar_to_RGB24(void *src, void *dest, int dest_stride, int width, int height, cudaStream_t stream)
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
    cudaStream_t stream)
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

} // extern "C"
