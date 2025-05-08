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

static __global__ void rgb24_to_yuv420_kernel(
    const uint8_t* __restrict__ rgb_in,
    uint8_t* __restrict__ y_plane,
    uint8_t* __restrict__ u_plane,
    uint8_t* __restrict__ v_plane,
    int width,
    int height,
    int rgb_stride,
    int y_stride,
    int uv_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int rgb_index = y * rgb_stride + x * 3;
    float R = static_cast<float>(rgb_in[rgb_index + 0]);
    float G = static_cast<float>(rgb_in[rgb_index + 1]);
    float B = static_cast<float>(rgb_in[rgb_index + 2]);

    // BT.709 limited-range RGB to YUV
    float Yf = 0.183f * R + 0.614f * G + 0.062f * B + 16.0f;
    //float Uf = -0.101f * R - 0.339f * G + 0.439f * B + 128.0f;
    //float Vf = 0.439f * R - 0.399f * G - 0.040f * B + 128.0f;

    // Store Y value
    int y_index = y * y_stride + x;
    y_plane[y_index] = clamp_u8(Yf);

    // Only store U and V for even rows and cols (4:2:0 subsampling)
    if ((x % 2 == 0) && (y % 2 == 0)) {
        // Average U and V over 2x2 block
        float Uacc = 0.0f;
        float Vacc = 0.0f;
        int count = 0;

        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int sx = x + dx;
                int sy = y + dy;
                if (sx < width && sy < height) {
                    int s_idx = sy * rgb_stride + sx * 3;
                    float r = static_cast<float>(rgb_in[s_idx + 0]);
                    float g = static_cast<float>(rgb_in[s_idx + 1]);
                    float b = static_cast<float>(rgb_in[s_idx + 2]);
                    Uacc += -0.101f * r - 0.339f * g + 0.439f * b + 128.0f;
                    Vacc +=  0.439f * r - 0.399f * g - 0.040f * b + 128.0f;
                    count++;
                }
            }
        }

        int uv_index = (y / 2) * uv_stride + (x / 2);
        u_plane[uv_index] = clamp_u8(Uacc / count);
        v_plane[uv_index] = clamp_u8(Vacc / count);
    }
}

void cuda_convertRGB24toYUV420(const uint8_t* d_rgb,
    int rgb_stride,
    uint8_t* d_y_plane,
    uint8_t* d_u_plane,
    uint8_t* d_v_plane,
    int y_stride,
    int uv_stride,
    int width,
    int height,
    cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);

    rgb24_to_yuv420_kernel<<<gridDim, blockDim, 0, stream>>>(
                d_rgb,
                d_y_plane, d_u_plane, d_v_plane,
                width, height,
                rgb_stride,
                y_stride,
                uv_stride);
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

static __global__ void rgb24_to_planar_fp32_kernel_stride_single_dest(
    const uint8_t* __restrict__ rgb24,
    float* __restrict__ dst,  // [R|G|B] planar FP32 interleaved in one buffer
    int width,
    int height,
    int stride  // input row stride in bytes
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;                 // index into each float plane
    int rgb_idx = y * stride + x * 3;    // index into packed uint8 RGB24

    float r = rgb24[rgb_idx    ] / 255.0f;
    float g = rgb24[rgb_idx + 1] / 255.0f;
    float b = rgb24[rgb_idx + 2] / 255.0f;

    int plane_size = width * height;
    dst[idx] = r;
    dst[idx + plane_size] = g;
    dst[idx + 2 * plane_size] = b;
}

void cuda_convert_rgb24_to_planar_fp32(
    const uint8_t* d_rgb24,  // input packed RGB24
    float* d_planar,         // output [R | G | B] FP32 buffer
    int width,
    int height,
    int stride,               // input stride in pixels (not bytes)
    cudaStream_t stream
)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rgb24_to_planar_fp32_kernel_stride_single_dest<<<grid, block, 0, stream>>>(
        d_rgb24, d_planar, width, height, stride
    );
}



} // extern "C"
