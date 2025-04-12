
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <jpeglib.h>
#include <setjmp.h>
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