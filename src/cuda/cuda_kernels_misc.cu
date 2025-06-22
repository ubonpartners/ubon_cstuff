#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <cassert>
#include "cuda.h"
#include "image.h"
#include "cuda_stuff.h"

#define CHECK_CUDA CHECK_CUDART_CALL

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

    assert((stride_a&3)==0);
    assert((stride_b&3)==0);

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
// Kernel: combined warp YUV -> RGB planar float (or fp16) per-pixel
__global__ void warpYUVToPlanarRGBKernel(
    cudaTextureObject_t *texY,
    cudaTextureObject_t *texU,
    cudaTextureObject_t *texV,
    const float *matrices,
    void *outPlanes,
    int outW, int outH,
    int batch,
    bool use_rgb24,
    bool use_fp16)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (x >= outW || y >= outH || b >= batch) return;

    // Load affine
    const float *M = matrices + b * 6;
    float sx = M[0] * x + M[1] * y + M[2];
    float sy = M[3] * x + M[4] * y + M[5];

    // Sample normalized [0,1]
    float y_norm = tex2D<float>(texY[b], sx, sy);
    float u_norm = tex2D<float>(texU[b], sx * 0.5f, sy * 0.5f);
    float v_norm = tex2D<float>(texV[b], sx * 0.5f, sy * 0.5f);

    // Normalize to standard range
    float Y = y_norm - (16.0f/255.0f);
    float U = u_norm - 0.5f;
    float V = v_norm - 0.5f;

    // BT.709 conversion
    float Rf = fminf(fmaxf(1.164f * Y + 1.793f * V, 0.0f), 1.0f);
    float Gf = fminf(fmaxf(1.164f * Y - 0.213f * U - 0.533f * V, 0.0f), 1.0f);
    float Bf = fminf(fmaxf(1.164f * Y + 2.112f * U, 0.0f), 1.0f);

    int planeSize = outW * outH;
    int base = b * 3 * planeSize;
    int idx = base + y * outW + x;

    if (use_rgb24)
    {
        uint8_t *out= (uint8_t*)outPlanes;
        out[idx*3+0]=(uint8_t)(Rf*255);
        out[idx*3+1]=(uint8_t)(Gf*255);
        out[idx*3+2]=(uint8_t)(Bf*255);
    } else if (use_fp16) {
        __half *outHf = (__half*)outPlanes;
        outHf[idx]               = __float2half(Rf);
        outHf[idx + planeSize]   = __float2half(Gf);
        outHf[idx + 2 * planeSize] = __float2half(Bf);
    } else {
        float *outF = (float*)outPlanes;
        outF[idx]               = Rf;
        outF[idx + planeSize]   = Gf;
        outF[idx + 2 * planeSize] = Bf;
    }
}

// Host function: batched warp of YUV to planar RGB
void cuda_warp_yuv420_to_planar_float(
    const image_t **inImgs,
    void *outPlanes,
    int batch,
    int outW,
    int outH,
    const float *h_matrices,
    bool use_rgb24,
    bool use_fp16,
    cudaStream_t stream)
{
    // Texture description (linear filtering of 8-bit as normalized floats)
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.addressMode[2]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;

    // Allocate host-side arrays
    cudaTextureObject_t *texY = (cudaTextureObject_t*)malloc(batch * sizeof(cudaTextureObject_t));
    cudaTextureObject_t *texU = (cudaTextureObject_t*)malloc(batch * sizeof(cudaTextureObject_t));
    cudaTextureObject_t *texV = (cudaTextureObject_t*)malloc(batch * sizeof(cudaTextureObject_t));
    if (!texY || !texU || !texV) {
        fprintf(stderr, "Failed to alloc texture object arrays\n");
        return;
    }

    cudaResourceDesc resDesc;
    for (int i = 0; i < batch; ++i) {
        assert(inImgs[i]->format==IMAGE_FORMAT_YUV420_DEVICE);
        // Y plane

        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType                 = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr      = inImgs[i]->y;
        resDesc.res.pitch2D.width       = inImgs[i]->width;
        resDesc.res.pitch2D.height      = inImgs[i]->height;
        resDesc.res.pitch2D.pitchInBytes= inImgs[i]->stride_y;
        resDesc.res.pitch2D.desc        = cudaCreateChannelDesc<unsigned char>();
        // Note; I find you get an error here if the y,u,v pointers are not
        // sufficiently aligned
        CHECK_CUDA(cudaCreateTextureObject(&texY[i], &resDesc, &texDesc, NULL));

        // U plane (half resolution)
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType                 = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr      = inImgs[i]->u;
        resDesc.res.pitch2D.width       = inImgs[i]->width/2;
        resDesc.res.pitch2D.height      = inImgs[i]->height/2;
        resDesc.res.pitch2D.pitchInBytes= inImgs[i]->stride_uv;
        resDesc.res.pitch2D.desc        = cudaCreateChannelDesc<unsigned char>();
        CHECK_CUDA(cudaCreateTextureObject(&texU[i], &resDesc, &texDesc, NULL));

        // V plane (half resolution)
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType                 = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr      = inImgs[i]->v;
        resDesc.res.pitch2D.width       = inImgs[i]->width/2;
        resDesc.res.pitch2D.height      = inImgs[i]->height/2;
        resDesc.res.pitch2D.pitchInBytes= inImgs[i]->stride_uv;
        resDesc.res.pitch2D.desc        = cudaCreateChannelDesc<unsigned char>();
        CHECK_CUDA(cudaCreateTextureObject(&texV[i], &resDesc, &texDesc, NULL));
    }
    // Copy to device
    cudaTextureObject_t *d_texY, *d_texU, *d_texV;
    CHECK_CUDA(cudaMalloc(&d_texY, batch * sizeof(cudaTextureObject_t)));
    CHECK_CUDA(cudaMalloc(&d_texU, batch * sizeof(cudaTextureObject_t)));
    CHECK_CUDA(cudaMalloc(&d_texV, batch * sizeof(cudaTextureObject_t)));
    CHECK_CUDA(cudaMemcpyAsync(d_texY, texY, batch * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_texU, texU, batch * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_texV, texV, batch * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice, stream));

    // Copy matrices
    float *d_matrices;
    CHECK_CUDA(cudaMalloc(&d_matrices, batch*6*sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(d_matrices, h_matrices, batch*6*sizeof(float), cudaMemcpyHostToDevice, stream));

    // Launch kernel
    dim3 block(16,16,1);
    dim3 grid((outW+15)/16, (outH+15)/16, batch);
    warpYUVToPlanarRGBKernel<<<grid, block, 0, stream>>>(
            d_texY, d_texU, d_texV,
            d_matrices, outPlanes,
            outW, outH, batch, use_rgb24, use_fp16);

    CHECK_CUDA(cudaGetLastError());
    // Cleanup

    cudaStreamSynchronize(stream);

    CHECK_CUDA(cudaFree(d_matrices));
    CHECK_CUDA(cudaFree(d_texY)); CHECK_CUDA(cudaFree(d_texU)); CHECK_CUDA(cudaFree(d_texV));
    for (int i = 0; i < batch; ++i) {
        CHECK_CUDA(cudaDestroyTextureObject(texY[i]));
        CHECK_CUDA(cudaDestroyTextureObject(texU[i]));
        CHECK_CUDA(cudaDestroyTextureObject(texV[i]));
    }
    free(texY); free(texU); free(texV);
}

__global__ void compute_motion_bytemask(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ U,
    const uint8_t* __restrict__ V,
    int stride_y,
    int stride_uv,
    float* __restrict__ noise_floor,
    int block_w,
    int block_h,
    float mad_delta,
    float alpha,
    float beta,
    uint8_t* __restrict__ row_masks)
{
    int tx = threadIdx.x;  // 0 .. bytes_per_row-1
    int y  = threadIdx.y;  // 0 .. block_h-1

    int bytes_per_row = (block_w + 7) >> 3;
    if (tx >= bytes_per_row || y >= block_h) return;

    int base_x = tx * 8;
    uint8_t byte_mask = 0;

    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        int x = base_x + bit;
        if (x < block_w) {
            int yy = y * 2;
            int xx = x * 2;
            uint8_t v0 = Y[xx     + (yy    ) * stride_y];
            uint8_t v1 = Y[xx + 1 + (yy    ) * stride_y];
            uint8_t v2 = Y[xx     + (yy+1) * stride_y];
            uint8_t v3 = Y[xx + 1 + (yy+1) * stride_y];
            uint8_t v4 = U[x + y * stride_uv];
            uint8_t v5 = V[x + y * stride_uv];

            uint8_t vmax = v0;
            vmax = vmax < v1 ? v1 : vmax;
            vmax = vmax < v2 ? v2 : vmax;
            vmax = vmax < v3 ? v3 : vmax;
            vmax = vmax < v4 ? v4 : vmax;
            vmax = vmax < v5 ? v5 : vmax;
            float fv = float(vmax);

            int idx = y * block_w + x;
            float nf = noise_floor[idx];
            float coeff = (fv < nf) ? alpha : beta;
            nf = nf * coeff + fv * (1.0f - coeff);
            noise_floor[idx] = nf;

            bool motion = (fv > nf + mad_delta);
            byte_mask |= static_cast<uint8_t>(motion ? (1u << (7-bit)) : 0u);
        }
    }

    row_masks[y * 8 + tx] = byte_mask;
}

void cuda_generate_motion_mask(
    const uint8_t* d_Y,
    const uint8_t* d_U,
    const uint8_t* d_V,
    int stride_y,
    int stride_uv,
    float* d_noise_floor,
    int block_w,
    int block_h,
    float mad_delta,
    float alpha,
    float beta,
    uint8_t* d_row_masks,
    cudaStream_t stream = 0)
{
    assert(block_w > 0 && block_w <= 64);
    assert(block_h > 0 && block_h <= 64);

    int bytes_per_row = (block_w + 7) >> 3;
    size_t total_bytes = size_t(block_h) * bytes_per_row;

    cudaError_t err = cudaMemsetAsync(d_row_masks, 0, total_bytes, stream);
    if (err != cudaSuccess) {
        // handle error
    }

    dim3 threads(bytes_per_row, block_h);
    compute_motion_bytemask<<<1, threads, 0, stream>>>(
        d_Y, d_U, d_V,
        stride_y, stride_uv,
        d_noise_floor,
        block_w, block_h,
        mad_delta,
        alpha, beta,
        d_row_masks);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // handle error
    }
}

} // extern "C"
