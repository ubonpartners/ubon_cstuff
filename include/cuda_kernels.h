#ifndef __CUDA_KERNELS_H
#define __CUDA_KERNELS_H

extern "C"
{
void cuda_convertYUVtoRGB_fp16(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
                     int y_stride, int uv_stride,
                     uint8_t *dest,
                     int width, int height, CUstream stream);

void cuda_convertYUVtoRGB_fp32(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
                     int y_stride, int uv_stride,
                     uint8_t *dest,
                     int width, int height, CUstream stream);

void cuda_half_to_float(void * d_input, void* h_output, int size);
void cuda_convert_fp16_planar_to_RGB24(void *src, void *dest, int dest_stride, int width, int height, CUstream stream);
void cuda_downsample_2x2(const uint8_t* d_src, int src_stride, uint8_t* d_dst, int dst_stride, int width, int height, CUstream stream);
void cuda_interleave_uv(const uint8_t* d_u, const uint8_t* d_v,
int src_stride_uv,
    uint8_t* d_dst,
    int dest_stride_uv,
    int width,
    int height,
    CUstream stream);
}

#endif
