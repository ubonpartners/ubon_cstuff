#ifndef __CUDA_KERNELS_H
#define __CUDA_KERNELS_H

extern "C"
{

void cuda_convertYUVtoRGB24(const uint8_t * d_y_plane, const uint8_t * d_u_plane, const uint8_t * d_v_plane,
                    int y_stride, int uv_stride,
                    uint8_t *dest, int dest_stride,
                    int width, int height, cudaStream_t stream);

void cuda_convertRGB24toYUV420(const uint8_t* d_rgb,int rgb_stride,
                        uint8_t* d_y_plane,uint8_t* d_u_plane,uint8_t* d_v_plane,
                        int y_stride,int uv_stride,
                        int width,int height,cudaStream_t stream);

void cuda_convert_rgb24_to_fp_planar(const uint8_t* rgb24,
                                    int rgb24_stride_bytes,
                                    int src_width,
                                    int src_height,
                                    void *dst_planar,
                                    int dst_width,
                                    int dst_plane_size,
                                    cudaStream_t stream,
                                    bool is_fp16);

void cuda_convert_yuv420_to_fp_planar(uint8_t * src_y, uint8_t *src_u, uint8_t *src_v,
                                    int src_stride_y, int src_stride_uv,
                                    void *dest, int dest_width, int dest_plane_size,
                                    int src_width, int src_height,
                                    cudaStream_t stream, bool is_fp16);

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
void cuda_fp_set(void* rgb_plane_ptr, int w, int h,int dst_w, int dst_plane_size_elements,cudaStream_t stream, bool is_fp16);
}
// nvidia code
void ResizeNv12(unsigned char *dpDstNv12, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char *dpSrcNv12, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char* dpDstNv12UV, cudaStream_t *pstream);

#endif
