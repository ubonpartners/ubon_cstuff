/*
 * This copyright notice applies to this header file only:
 *
 * Copyright (c) 2010-2024 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the software, and to permit persons to whom the
 * software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

 #include <cuda_runtime.h>
#include <stdint.h>

#define ck(x) x
 template<typename YuvUnitx2>
 static __global__ void Resize(cudaTextureObject_t texY, cudaTextureObject_t texUv,
         uint8_t *pDst, uint8_t *pDstUV, int nPitch, int nWidth, int nHeight,
         float fxScale, float fyScale)
 {
     int ix = blockIdx.x * blockDim.x + threadIdx.x,
         iy = blockIdx.y * blockDim.y + threadIdx.y;

     if (ix >= nWidth / 2 || iy >= nHeight / 2) {
         return;
     }
     float offsx=0.5;
     float offsy=0.5;
     float offsx_uv=0.0;
     float offsy_uv=0.5;
     int x = ix * 2, y = iy * 2;
     typedef decltype(YuvUnitx2::x) YuvUnit;
     const int MAX = (1 << (sizeof(YuvUnit) * 8)) - 1;
     *(YuvUnitx2 *)(pDst + y * nPitch + x * sizeof(YuvUnit)) = YuvUnitx2 {
         (YuvUnit)(tex2D<float>(texY, x / fxScale+ offsx, y / fyScale+ offsy) * MAX),
         (YuvUnit)(tex2D<float>(texY, (x + 1) / fxScale+ offsx, y / fyScale+ offsy) * MAX)
     };
     y++;
     *(YuvUnitx2 *)(pDst + y * nPitch + x * sizeof(YuvUnit)) = YuvUnitx2 {
         (YuvUnit)(tex2D<float>(texY, x / fxScale+ offsx, y / fyScale+ offsy) * MAX),
         (YuvUnit)(tex2D<float>(texY, (x + 1) / fxScale+offsx, y / fyScale+ offsy) * MAX)
     };
     float2 uv = tex2D<float2>(texUv, ix / fxScale+offsx_uv, (nHeight + iy) / fyScale + offsy_uv);
     *(YuvUnitx2 *)(pDstUV + iy * nPitch + ix * 2 * sizeof(YuvUnit)) = YuvUnitx2{ (YuvUnit)(uv.x * MAX), (YuvUnit)(uv.y * MAX) };
 }

 template <typename YuvUnitx2>
 static void Resize(unsigned char *dpDst,
    unsigned char* dpDstUV, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, cudaStream_t *pstream) {
     cudaResourceDesc resDesc = {};
     resDesc.resType = cudaResourceTypePitch2D;
     resDesc.res.pitch2D.devPtr = dpSrc;
     resDesc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(YuvUnitx2::x)>();
     resDesc.res.pitch2D.width = nSrcWidth;
     resDesc.res.pitch2D.height = nSrcHeight;
     resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

     cudaTextureDesc texDesc = {};
     texDesc.filterMode = cudaFilterModeLinear;
     texDesc.readMode = cudaReadModeNormalizedFloat;

     cudaTextureObject_t texY=0;
     ck(cudaCreateTextureObject(&texY, &resDesc, &texDesc, NULL));

     resDesc.res.pitch2D.desc = cudaCreateChannelDesc<YuvUnitx2>();
     resDesc.res.pitch2D.width = nSrcWidth / 2;
     resDesc.res.pitch2D.height = nSrcHeight * 3 / 2;

     cudaTextureObject_t texUv=0;
     ck(cudaCreateTextureObject(&texUv, &resDesc, &texDesc, NULL));

     Resize<YuvUnitx2> << <dim3((nDstWidth + 31) / 32, (nDstHeight + 31) / 32), dim3(16, 16), 0, *pstream >> >(texY, texUv, dpDst, dpDstUV,
         nDstPitch, nDstWidth, nDstHeight, 1.0f * nDstWidth / nSrcWidth, 1.0f * nDstHeight / nSrcHeight);

     ck(cudaDestroyTextureObject(texY));
     ck(cudaDestroyTextureObject(texUv));
 }

 void ResizeNv12(unsigned char *dpDstNv12, int nDstPitch, int nDstWidth, int nDstHeight,
    unsigned char *dpSrcNv12, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char* dpDstNv12UV, cudaStream_t *pstream)
 {
     unsigned char* dpDstUV = dpDstNv12UV ? dpDstNv12UV : dpDstNv12 + (nDstPitch*nDstHeight);
     return Resize<uchar2>(dpDstNv12, dpDstUV, nDstPitch, nDstWidth, nDstHeight, dpSrcNv12, nSrcPitch, nSrcWidth, nSrcHeight, pstream);
 }


 void ResizeP016(unsigned char *dpDstP016, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char *dpSrcP016, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char* dpDstP016UV, cudaStream_t *pstream)
 {
     unsigned char* dpDstUV = dpDstP016UV ? dpDstP016UV : dpDstP016 + (nDstPitch*nDstHeight);
     return Resize<ushort2>(dpDstP016, dpDstUV, nDstPitch, nDstWidth, nDstHeight, dpSrcP016, nSrcPitch, nSrcWidth, nSrcHeight, pstream);
 }

 static __global__ void Scale(cudaTextureObject_t texSrc,
     uint8_t *pDst, int nPitch, int nWidth, int nHeight,
     float fxScale, float fyScale)
 {
     int x = blockIdx.x * blockDim.x + threadIdx.x,
         y = blockIdx.y * blockDim.y + threadIdx.y;

     if (x >= nWidth || y >= nHeight)
     {
         return;
     }

     *(unsigned char*)(pDst + (y * nPitch) + x) = (unsigned char)(fminf((tex2D<float>(texSrc, x * fxScale, y * fyScale)) * 255.0f, 255.0f));
 }

 static __global__ void Scale_uv(cudaTextureObject_t texSrc,
     uint8_t *pDst, int nPitch, int nWidth, int nHeight,
     float fxScale, float fyScale)
 {
     int x = blockIdx.x * blockDim.x + threadIdx.x,
         y = blockIdx.y * blockDim.y + threadIdx.y;

     if (x >= nWidth || y >= nHeight)
     {
         return;
     }

     float2 uv = tex2D<float2>(texSrc, x * fxScale, y * fyScale);
     uchar2 uvOut = uchar2{ (unsigned char)(fminf(uv.x * 255.0f, 255.0f)), (unsigned char)(fminf(uv.y * 255.0f, 255.0f)) };

     *(uchar2*)(pDst + (y * nPitch) + 2 * x) = uvOut;
 }

 void ScaleKernelLaunch(unsigned char *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, bool bUVPlane = false)
 {
     cudaResourceDesc resDesc = {};
     resDesc.resType = cudaResourceTypePitch2D;
     resDesc.res.pitch2D.devPtr = dpSrc;
     resDesc.res.pitch2D.desc = bUVPlane ? cudaCreateChannelDesc<uchar2>() : cudaCreateChannelDesc<unsigned char>();
     resDesc.res.pitch2D.width = nSrcWidth;
     resDesc.res.pitch2D.height = nSrcHeight;
     resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

     cudaTextureDesc texDesc = {};
     texDesc.filterMode = cudaFilterModeLinear;
     texDesc.readMode = cudaReadModeNormalizedFloat;

     texDesc.addressMode[0] = cudaAddressModeClamp;
     texDesc.addressMode[1] = cudaAddressModeClamp;
     texDesc.addressMode[2] = cudaAddressModeClamp;

     cudaTextureObject_t texSrc = 0;
     ck(cudaCreateTextureObject(&texSrc, &resDesc, &texDesc, NULL));

     dim3 blockSize(16, 16, 1);
     dim3 gridSize(((uint32_t)nDstWidth + blockSize.x - 1) / blockSize.x, ((uint32_t)nDstHeight + blockSize.y - 1) / blockSize.y, 1);

     if (bUVPlane)
     {
         Scale_uv << <gridSize, blockSize >> >(texSrc, dpDst,
             nDstPitch, nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight);
     }
     else
     {
         Scale << <gridSize, blockSize >> >(texSrc, dpDst,
             nDstPitch, nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight);
     }

     ck(cudaGetLastError());
     ck(cudaDestroyTextureObject(texSrc));
 }

 void ScaleYUV420(unsigned char *dpDstY,
                  unsigned char* dpDstU,
                 unsigned char* dpDstV,
                 int nDstPitch,
                 int nDstChromaPitch,
                 int nDstWidth,
                 int nDstHeight,
                 unsigned char *dpSrcY,
                 unsigned char* dpSrcU,
                 unsigned char* dpSrcV,
                 int nSrcPitch,
                 int nSrcChromaPitch,
                 int nSrcWidth,
                 int nSrcHeight,
                 bool bSemiplanar)
 {
     int chromaWidthDst = (nDstWidth + 1) / 2;
     int chromaHeightDst = (nDstHeight + 1) / 2;

     int chromaWidthSrc = (nSrcWidth + 1) / 2;
     int chromaHeightSrc = (nSrcHeight + 1) / 2;

     ScaleKernelLaunch(dpDstY, nDstPitch, nDstWidth, nDstHeight, dpSrcY, nSrcPitch, nSrcWidth, nSrcHeight);

     if (bSemiplanar)
     {
         ScaleKernelLaunch(dpDstU, nDstChromaPitch, chromaWidthDst, chromaHeightDst, dpSrcU, nSrcChromaPitch, chromaWidthSrc, chromaHeightSrc, true);
     }
     else
     {
         ScaleKernelLaunch(dpDstU, nDstChromaPitch, chromaWidthDst, chromaHeightDst, dpSrcU, nSrcChromaPitch, chromaWidthSrc, chromaHeightSrc);
         ScaleKernelLaunch(dpDstV, nDstChromaPitch, chromaWidthDst, chromaHeightDst, dpSrcV, nSrcChromaPitch, chromaWidthSrc, chromaHeightSrc);
     }
 }


 /**
  * CUDA kernel for copying a portion of the Y plane
  */
 __global__ void cropYPlaneKernel(
     const uint8_t* src,
     uint8_t* dst,
     int src_width,
     int dst_width,
     int dst_height,
     int crop_x,
     int crop_y
 ) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;

     if (x < dst_width && y < dst_height) {
         // Calculate source and destination offsets
         int src_offset = (y + crop_y) * src_width + (x + crop_x);
         int dst_offset = y * dst_width + x;

         // Copy Y value
         dst[dst_offset] = src[src_offset];
     }
 }

 /**
  * CUDA kernel for copying a portion of the UV plane
  */
 __global__ void cropUVPlaneKernel(
     const uint8_t* src,
     uint8_t* dst,
     int src_width,
     int dst_width,
     int dst_height,
     int crop_x,
     int crop_y,
     int src_y_plane_size,
     int dst_y_plane_size
 ) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;

     // UV plane has half the height of the Y plane in NV12
     if (x < dst_width && y < dst_height / 2) {
         // Calculate source and destination offsets
         // NV12 format has interleaved UV values, so we need to adjust accordingly
         // Each UV pair serves a 2x2 block of pixels
         int src_offset = src_y_plane_size + ((y + crop_y / 2) * src_width + (crop_x + x * 2)) / 2 * 2;
         int dst_offset = dst_y_plane_size + y * dst_width + x * 2;

         // Copy UV pair (2 bytes)
         dst[dst_offset] = src[src_offset];
         if (x * 2 + 1 < dst_width) {
             dst[dst_offset + 1] = src[src_offset + 1];
         }
     }
 }

 /**
  * Crop an NV12 image in CUDA device memory
  *
  * @param d_src           Source NV12 image in CUDA device memory
  * @param src_width       Width of source image
  * @param src_height      Height of source image
  * @param d_dst           Destination buffer in CUDA device memory (must be pre-allocated)
  * @param crop_x          X-coordinate of top-left corner of crop region
  * @param crop_y          Y-coordinate of top-left corner of crop region
  * @param crop_width      Width of crop region
  * @param crop_height     Height of crop region
  * @return                cudaSuccess if successful, error code otherwise
  */
 cudaError_t cropNV12CUDA(
     const uint8_t* d_src,
     int src_width,
     int src_height,
     uint8_t* d_dst,
     int crop_x,
     int crop_y,
     int crop_width,
     int crop_height
 ) {
     cudaError_t cudaStatus;

     // Validate parameters
     if (crop_x < 0 || crop_y < 0 ||
         crop_width <= 0 || crop_height <= 0 ||
         crop_x + crop_width > src_width ||
         crop_y + crop_height > src_height) {
         return cudaErrorInvalidValue;
     }

     // Calculate plane sizes
     int src_y_plane_size = src_width * src_height;
     int dst_y_plane_size = crop_width * crop_height;

     // Define block and grid dimensions
     dim3 blockSize(16, 16);
     dim3 gridSizeY((crop_width + blockSize.x - 1) / blockSize.x,
                   (crop_height + blockSize.y - 1) / blockSize.y);

     // Launch kernel to copy Y plane
     cropYPlaneKernel<<<gridSizeY, blockSize>>>(
         d_src, d_dst, src_width, crop_width, crop_height, crop_x, crop_y
     );

     // Check for errors
     cudaStatus = cudaGetLastError();
     if (cudaStatus != cudaSuccess) {
         return cudaStatus;
     }

     // Define grid dimensions for UV plane (half height)
     dim3 gridSizeUV((crop_width + blockSize.x - 1) / blockSize.x,
                    (crop_height/2 + blockSize.y - 1) / blockSize.y);

     // Launch kernel to copy UV plane
     cropUVPlaneKernel<<<gridSizeUV, blockSize>>>(
         d_src, d_dst, src_width, crop_width, crop_height, crop_x, crop_y,
         src_y_plane_size, dst_y_plane_size
     );

     // Check for errors
     cudaStatus = cudaGetLastError();
     if (cudaStatus != cudaSuccess) {
         return cudaStatus;
     }

     // Wait for completion
     return cudaDeviceSynchronize();
 }
