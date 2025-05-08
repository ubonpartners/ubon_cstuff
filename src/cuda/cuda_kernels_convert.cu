#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "cuda.h"

extern "C" {

static __device__ inline float clamp(float x) {
    return x < 0 ? 0 : (x > 1 ? 1 : x);
}

static __device__ uint8_t clamp_u8(float val) {
    return static_cast<uint8_t>(fminf(fmaxf(val+0.5f, 0.0f), 255.0f));
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

// fixed bilinear weights for offsets ±0.25
__constant__ float w00 = 0.5625f;  // (1-0.25)*(1-0.25)
__constant__ float w10 = 0.1875f;  // 0.25*(1-0.25)
__constant__ float w01 = 0.1875f;  // (1-0.25)*0.25
__constant__ float w11 = 0.0625f;  // 0.25*0.25

// One thread per UV sample → process a 2×2 block of Y
static __global__ void yuv420_to_rgb24_unroll2x2(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ u_plane,
    const uint8_t* __restrict__ v_plane,
    uint8_t* __restrict__ rgb_out,
    int width, int height,
    int y_stride, int uv_stride, int rgb_stride)
{
    // UV coordinates
    int ux = blockIdx.x * blockDim.x + threadIdx.x;
    int uy = blockIdx.y * blockDim.y + threadIdx.y;
    int halfW = width  / 2;
    int halfH = height / 2;
    if (ux >= halfW || uy >= halfH) return;

    // clamp neighbor indices
    int nx = min(ux + 1, halfW - 1);
    int ny = min(uy + 1, halfH - 1);

    // fetch the 4 raw U/V values
    int p00 = uy * uv_stride + ux;
    int p10 = uy * uv_stride + nx;
    int p01 = ny * uv_stride + ux;
    int p11 = ny * uv_stride + nx;

    float u00 = float(u_plane[p00]) - 128.0f;
    float u10 = float(u_plane[p10]) - 128.0f;
    float u01 = float(u_plane[p01]) - 128.0f;
    float u11 = float(u_plane[p11]) - 128.0f;

    float v00 = float(v_plane[p00]) - 128.0f;
    float v10 = float(v_plane[p10]) - 128.0f;
    float v01 = float(v_plane[p01]) - 128.0f;
    float v11 = float(v_plane[p11]) - 128.0f;

    // precompute the four interpolated U/V for the 2×2 luma block
    float u00p = w00*u00 + w10*u10 + w01*u01 + w11*u11;
    float u10p = w10*u00 + w00*u10 + w11*u01 + w01*u11;
    float u01p = w01*u00 + w11*u10 + w00*u01 + w10*u11;
    float u11p = w11*u00 + w01*u10 + w10*u01 + w00*u11;

    float v00p = w00*v00 + w10*v10 + w01*v01 + w11*v11;
    float v10p = w10*v00 + w00*v10 + w11*v01 + w01*v11;
    float v01p = w01*v00 + w11*v10 + w00*v01 + w10*v11;
    float v11p = w11*v00 + w01*v10 + w10*v01 + w00*v11;

    // Y block origin
    int x0 = ux * 2;
    int y0 = uy * 2;

    // load the four Y samples
    int idxY00 = y0     * y_stride + x0;
    int idxY01 = idxY00 + 1;
    int idxY10 = (y0+1) * y_stride + x0;
    int idxY11 = idxY10 + 1;

    float c00 = float(y_plane[idxY00]) - 16.0f;
    float c01 = float(y_plane[idxY01]) - 16.0f;
    float c10 = float(y_plane[idxY10]) - 16.0f;
    float c11 = float(y_plane[idxY11]) - 16.0f;

    // convert & pack helper
    auto conv = [&](float c, float up, float vp){
        float rf = 1.164f * c + 1.793f * vp;
        float gf = 1.164f * c - 0.213f * up  - 0.533f * vp;
        float bf = 1.164f * c + 2.112f * up;
        return make_uchar3(clamp_u8(rf), clamp_u8(gf), clamp_u8(bf));
    };

    uchar3 rgb00 = conv(c00, u00p, v00p);
    uchar3 rgb01 = conv(c01, u10p, v10p);
    uchar3 rgb10 = conv(c10, u01p, v01p);
    uchar3 rgb11 = conv(c11, u11p, v11p);

    // write out
    int outRow0 = y0     * rgb_stride + x0*3;
    int outRow1 = (y0+1) * rgb_stride + x0*3;

    uint8_t* ptr00 = rgb_out + outRow0;
    uint8_t* ptr01 = ptr00 + 3;
    uint8_t* ptr10 = rgb_out + outRow1;
    uint8_t* ptr11 = ptr10 + 3;

    ptr00[0]=rgb00.x; ptr00[1]=rgb00.y; ptr00[2]=rgb00.z;
    ptr01[0]=rgb01.x; ptr01[1]=rgb01.y; ptr01[2]=rgb01.z;
    ptr10[0]=rgb10.x; ptr10[1]=rgb10.y; ptr10[2]=rgb10.z;
    ptr11[0]=rgb11.x; ptr11[1]=rgb11.y; ptr11[2]=rgb11.z;
}

void cuda_convertYUVtoRGB24(
    const uint8_t *d_y, const uint8_t *d_u, const uint8_t *d_v,
    int y_stride, int uv_stride,
    uint8_t *d_rgb, int rgb_stride,
    int width, int height,
    cudaStream_t stream)
{
    dim3 block(16,16);
    dim3 grid((width/2  + block.x-1)/block.x,
              (height/2 + block.y-1)/block.y);

    yuv420_to_rgb24_unroll2x2<<<grid,block,0,stream>>>(
        d_y, d_u, d_v, d_rgb,
        width, height,
        y_stride, uv_stride, rgb_stride);
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
