
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <cassert>
#include "image.h"

struct my_error_mgr
{
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

typedef struct my_error_mgr *my_error_ptr;

void my_error_exit(j_common_ptr cinfo)
{
    my_error_ptr myerr = (my_error_ptr) cinfo->err;
    longjmp(myerr->setjmp_buffer, 1);
}

image_t *decode_jpeg(uint8_t *buffer, size_t size)
{
    // Decode JPEG using libjpeg
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer))
    {
        jpeg_destroy_decompress(&cinfo);
        return 0;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, buffer, size);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    int row_stride = cinfo.output_width * cinfo.output_components;
    image_t *img=create_image(cinfo.output_width, cinfo.output_height, IMAGE_FORMAT_RGB24_HOST);

    while (cinfo.output_scanline < cinfo.output_height)
    {
        unsigned char *buffer_array[1];
        buffer_array[0] = img->rgb + cinfo.output_scanline * img->stride_rgb;
        jpeg_read_scanlines(&cinfo, buffer_array, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return img;
}

image_t *load_jpeg(const char *file)
{
    FILE *f=fopen(file, "rb");
    if (!f) return 0;

    fseek(f, 0L, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0L, SEEK_SET);
    image_t *ret=0;
    uint8_t *mem=(uint8_t*)malloc(sz);
    if (mem)
    {
        fread(mem, 1, sz, f);
        ret=decode_jpeg(mem, sz);
        free(mem);
    }
    return ret;
}

static int save_jpeg_yuv420(const char *filename, image_t *img, int quality)
{
    if (!img || img->format != IMAGE_FORMAT_YUV420_HOST)
        return -1;

    image_sync(img);

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *outfile = fopen(filename, "wb");
    if (!outfile)
        return -1;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = img->width;
    cinfo.image_height = img->height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_YCbCr;

    jpeg_set_defaults(&cinfo);
    cinfo.raw_data_in = TRUE;
    cinfo.jpeg_color_space = JCS_YCbCr;
    jpeg_set_quality(&cinfo, quality, TRUE);

    // Set 4:2:0 sampling factors manually
    cinfo.comp_info[0].h_samp_factor = 2;
    cinfo.comp_info[0].v_samp_factor = 2;
    cinfo.comp_info[1].h_samp_factor = 1;
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[2].h_samp_factor = 1;
    cinfo.comp_info[2].v_samp_factor = 1;

    jpeg_start_compress(&cinfo, TRUE);

    int y_stride = img->stride_y;
    int uv_stride = img->stride_uv;
    int width = img->width;
    int height = img->height;

    // Allocate row pointers dynamically
    JSAMPARRAY y_rows = (JSAMPARRAY)malloc(sizeof(JSAMPROW) * 16);
    JSAMPARRAY u_rows = (JSAMPARRAY)malloc(sizeof(JSAMPROW) * 8);
    JSAMPARRAY v_rows = (JSAMPARRAY)malloc(sizeof(JSAMPROW) * 8);
    JSAMPIMAGE planes = (JSAMPIMAGE)malloc(sizeof(JSAMPARRAY) * 3);

    planes[0] = y_rows;
    planes[1] = u_rows;
    planes[2] = v_rows;

    while (cinfo.next_scanline < cinfo.image_height) {
        int lines = 16;
        if (cinfo.next_scanline + lines > height)
            lines = height - cinfo.next_scanline;

        for (int i = 0; i < lines; i++)
            y_rows[i] = img->y + (cinfo.next_scanline + i) * y_stride;

        for (int i = 0; i < lines / 2; i++) {
            int uv_row = (cinfo.next_scanline / 2) + i;
            u_rows[i] = img->u + uv_row * uv_stride;
            v_rows[i] = img->v + uv_row * uv_stride;
        }

        jpeg_write_raw_data(&cinfo, planes, lines);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);

    free(y_rows);
    free(u_rows);
    free(v_rows);
    free(planes);

    return 0;
}

static int save_jpeg_rgb24(const char *filename, image_t *img, int quality)
{
    if (!img || img->format != IMAGE_FORMAT_RGB24_HOST)
        return -1;

    image_sync(img);

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *outfile = fopen(filename, "wb");
    if (!outfile)
        return -1;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = img->width;
    cinfo.image_height = img->height;
    cinfo.input_components = 3; // RGB
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height)
    {
        JSAMPROW row_pointer[1];
        row_pointer[0] = img->rgb + cinfo.next_scanline * img->stride_rgb;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);

    return 0;
}

void save_jpeg(const char *filename, image_t *img)
{
    if (!img) return;
    image_format_t inter=img->format;
    if (img->format==IMAGE_FORMAT_RGB24_HOST)
        save_jpeg_rgb24(filename, img, 90);
    else if (img->format==IMAGE_FORMAT_YUV420_HOST)
        save_jpeg_yuv420(filename, img, 90);
    else if ( (img->format==IMAGE_FORMAT_YUV420_DEVICE)
            ||(img->format==IMAGE_FORMAT_NV12_DEVICE)
            ||(img->format==IMAGE_FORMAT_MONO_DEVICE))
        inter=IMAGE_FORMAT_YUV420_HOST;
    else
    {
        inter=IMAGE_FORMAT_RGB24_HOST;
    }

    image_t *temp=image_convert(img, inter);
    save_jpeg(filename, temp);
    destroy_image(temp);
}