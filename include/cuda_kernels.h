#ifndef __CUDA_KERNELS_H
#define __CUDA_KERNELS_H

extern "C"
{
void cuda_convertYUVtoRGB_fp16(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
                     int y_stride, int uv_stride,
                     uint8_t *dest, int plane_offset,
                     int src_width, int src_height,
                     int dst_width, int dst_height,
                     cudaStream_t stream);

void cuda_convertYUVtoRGB_fp32(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
                     int y_stride, int uv_stride,
                     uint8_t *dest, int plane_offset,
                     int src_width, int src_height,
                     int dst_width, int dst_height,
                     cudaStream_t stream);
void cuda_convertYUVtoRGB24(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
                    int y_stride, int uv_stride,
                    uint8_t *dest, int dest_stride,
                    int width, int height, cudaStream_t stream);
void cuda_convertRGB24toYUV420(const uint8_t* d_rgb,int rgb_stride,
                        uint8_t* d_y_plane,uint8_t* d_u_plane,uint8_t* d_v_plane,
                        int y_stride,int uv_stride,
                        int width,int height,cudaStream_t stream);
void cuda_convert_rgb24_to_planar_fp32(
                    const uint8_t* d_rgb24,  // input packed RGB24
                    float* d_planar,         // output [R | G | B] FP32 buffer
                    int src_width,
                    int src_height,
                    int dst_width,
                    int dst_height,
                    int stride,               // input stride in pixels (not bytes)
                    cudaStream_t stream);
void cuda_convert_rgb24_to_planar_fp16(
                    const uint8_t* d_rgb24,  // input packed RGB24
                    void* d_planar,         // output [R | G | B] FP16 buffer
                    int src_width,
                    int src_height,
                    int dst_width,
                    int dst_height,
                    int stride,               // input stride in pixels (not bytes)
                    cudaStream_t stream);
void cuda_half_to_float(void * d_input, void* h_output, int size);
void cuda_convert_fp16_planar_to_RGB24(void *src, void *dest, int dest_stride, int width, int height, cudaStream_t stream);
void cuda_convert_fp32_planar_to_RGB24(void *src, void *dest, int dest_stride, int width, int height, cudaStream_t stream);
void cuda_downsample_2x2(const uint8_t* d_src, int src_stride, uint8_t* d_dst, int dst_stride, int width, int height, cudaStream_t stream);
void cuda_interleave_uv(const uint8_t* d_u, const uint8_t* d_v,
int src_stride_uv,
    uint8_t* d_dst,
    int dest_stride_uv,
    int width,
    int height,
    cudaStream_t stream);
void cuda_hash_2d(const uint8_t* d_data, int w, int h, int stride, uint32_t *dest, cudaStream_t stream);
void compute_4x4_mad_mask(uint8_t *a, int stride_a, uint8_t *b, int stride_b,
uint8_t *out, int stride_out, int width, int height, cudaStream_t stream);

}

// nvidia code
void ResizeNv12(unsigned char *dpDstNv12, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char *dpSrcNv12, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char* dpDstNv12UV, cudaStream_t *pstream);

#endif
