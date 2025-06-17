#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <mutex>
#include <cassert>
#include "image.h"
#include "log.h"
#include "profile.h"

#define MAX_IMAGE_DIMENSION 10000

struct my_error_mgr {
    struct jpeg_error_mgr pub;    /* “public” fields */
    jmp_buf setjmp_buffer;        /* for return to caller */
};
typedef struct my_error_mgr * my_error_ptr;

/* Replacement for stderr output */
void my_output_message(j_common_ptr cinfo) {
    /* swallow or route into your log system */
    char buffer[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buffer);
    log_error("libjpeg warning: %s", buffer);
}

/* Replacement for exit() on fatal errors */
void my_error_exit(j_common_ptr cinfo) {
    my_error_ptr err = (my_error_ptr)cinfo->err;
    /* Clean up libjpeg state */
    jpeg_destroy_decompress((j_decompress_ptr)cinfo);
    /* Jump back to caller */
    longjmp(err->setjmp_buffer, 1);
}

image_t *decode_jpeg(uint8_t *buffer, size_t size) {
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    image_t *img = NULL;

    /* 1) Install our error handlers _before_ any libjpeg call */
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    jerr.pub.output_message = my_output_message;

    /* 2) Establish setjmp “checkpoint” */
    if (setjmp(jerr.setjmp_buffer)) {
        /* If we get here, the JPEG code signaled an error. */
        if (img) {
            destroy_image(img);
            img=0;
        }
        return 0;
    }

    /* 3) Now it’s safe to call libjpeg routines */
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, buffer, size);

    /* 4) Read header and sanity‐check dimensions */
    jpeg_read_header(&cinfo, TRUE);
    if (cinfo.image_width  > MAX_IMAGE_DIMENSION ||
        cinfo.image_height > MAX_IMAGE_DIMENSION) {
        jpeg_destroy_decompress(&cinfo);
        log_error("JPEG dimensions too large: %ux%u",
                  cinfo.image_width, cinfo.image_height);
        return 0;
    }

    /* --- downscale early if user’s max is smaller --- */
    int max_w=1920;
    int max_h=1080;
    if (cinfo.image_width > max_w || cinfo.image_height > max_h) {
        int denom = 1;
        while (denom < 8
               && (cinfo.image_width  / (denom * 2) > max_w
                || cinfo.image_height / (denom * 2) > max_h)) {
            denom *= 2;
        }
        cinfo.scale_num   = 1;
        cinfo.scale_denom = denom;
    }
    if (cinfo.scale_denom!=1)
    {
        log_info("Downscaling jpeg %dx%d by %d",cinfo.image_width,cinfo.image_height,cinfo.scale_denom);
    }


    cinfo.dct_method = JDCT_IFAST;
    if (cinfo.jpeg_color_space == JCS_CMYK || cinfo.jpeg_color_space == JCS_YCCK) {
        cinfo.out_color_space = JCS_CMYK;
    } else {
        cinfo.out_color_space = JCS_RGB;
    }
    jpeg_start_decompress(&cinfo);

    /* 5) Allocate only after header is validated */
    img = create_image(cinfo.output_width,
                       cinfo.output_height,
                       IMAGE_FORMAT_RGB24_HOST);
    if (!img) {
        jpeg_destroy_decompress(&cinfo);
        log_error("Out of memory allocating image");
        return 0;
    }

    /* 6) Read scanlines */

    if ((cinfo.out_color_space == JCS_CMYK)||(cinfo.out_color_space == JCS_YCCK))
    {
        uint8_t *rowbuf=(uint8_t *)malloc(cinfo.output_width*cinfo.output_components);
        uint8_t *dst = img->rgb;
        assert(rowbuf!=0);
        JSAMPROW ptr = rowbuf;
        while (cinfo.output_scanline < cinfo.output_height) {
            jpeg_read_scanlines(&cinfo, &ptr, 1);
            for(int i=0;i<cinfo.output_width;i++)
            {
                float C=1.0-rowbuf[i*4+0]/255.0;
                float M=1.0-rowbuf[i*4+1]/255.0;
                float Y=1.0-rowbuf[i*4+2]/255.0;
                float K=1.0-rowbuf[i*4+3]/255.0;
                dst[3*i+0]=(uint8_t)((1.0-C)*(1.0-K)*255+0.5f);
                dst[3*i+1]=(uint8_t)((1.0-M)*(1.0-K)*255+0.5f);
                dst[3*i+2]=(uint8_t)((1.0-Y)*(1.0-K)*255+0.5f);
            }
            dst+=img->stride_rgb;
        }
        free(rowbuf);
    }
    else
    {
        assert(cinfo.output_width*cinfo.output_components<=img->stride_rgb);
        while (cinfo.output_scanline < cinfo.output_height) {
            JSAMPROW row_pointer = img->rgb
            + cinfo.output_scanline * img->stride_rgb;
            jpeg_read_scanlines(&cinfo, &row_pointer, 1);
        }
    }

    /* 7) Finish up cleanly */
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return img;
}

#define NVJPEG_CHECK(status) do { \
    nvjpegStatus_t err = status; \
    if (err != NVJPEG_STATUS_SUCCESS) { \
        std::cerr << "nvJPEG error: " << err << " at line " << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)


static nvjpegHandle_t nvjpeg_handle = nullptr;
static nvjpegJpegState_t nvjpeg_state = nullptr;
static std::mutex nvjpeg_mutex;
static bool nvjpeg_inited=false;

static void init_nvjpeg()
{
    std::lock_guard<std::mutex> lock(nvjpeg_mutex);
    if (!nvjpeg_inited)
    {
        NVJPEG_CHECK(nvjpegCreateSimple(&nvjpeg_handle));
        NVJPEG_CHECK(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
        nvjpeg_inited=true;
    }
}

image_t *decode_jpeg_nvjpeg(uint8_t *buffer, size_t size)
{
    init_nvjpeg();

    nvjpegStatus_t status;

    std::lock_guard<std::mutex> lock(nvjpeg_mutex);
    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling;

    nvjpegImage_t nv_image={0};
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};

    nvjpegOutputFormat_t out_fmt = NVJPEG_OUTPUT_RGBI;
    status=nvjpegGetImageInfo(nvjpeg_handle, buffer, size,
                              &nComponents,
                              &subsampling, widths, heights);

    if (status != NVJPEG_STATUS_SUCCESS)
    {
        log_error("NVJPEG GetImageInfo failed");
        return 0;
    }

    int width = widths[0];
    int height = heights[0];

    if (width  > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
        log_error("NVEC JPEG dimensions too large: %ux%u", width, height);
        return 0;
    }

    image_t *img=create_image(width, height, IMAGE_FORMAT_RGB24_DEVICE);

    nv_image.channel[0] = img->rgb;
    nv_image.pitch[0] = img->stride_rgb;

    // Decode to RGB interleaved
    status=nvjpegDecode(nvjpeg_handle, nvjpeg_state,buffer, size, out_fmt, &nv_image, img->stream);
    if (status != NVJPEG_STATUS_SUCCESS)
    {
        //log_error("NVJPEG nvjpegDecode failed");
        destroy_image(img);
        return 0;
    }
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(img->stream); // TODO: should not be needed but seems to be
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    log_error("NVJPEG CUDA error: %s", cudaGetErrorString(err));
    }
    return img;
}

image_t *load_jpeg(const char *file)
{
    FILE *f=fopen(file, "rb");
    if (!f)
    {
        log_error("Could not open file %s",file);
        return 0;
    }
    //double t=profile_time();
    fseek(f, 0L, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0L, SEEK_SET);
    image_t *ret=0;
    uint8_t *mem=(uint8_t*)malloc(sz);
    if (mem)
    {
        fread(mem, 1, sz, f);
        ret=decode_jpeg_nvjpeg(mem,sz);
        if (ret==0)
        {
            //log_error("NVJPEG could not decode jpeg %s", file);
            // failed; try libjpeg (e.g. nvjpeg can't seem to code CMMY)
            ret=decode_jpeg(mem, sz);
            if (ret==0)
            {
                log_error("Could not decode jpeg %s", file);
            }
        }
        free(mem);
    }
    fclose(f);
    if (ret==0) log_error("Failed to decode jpeg %s",file);;
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

static int has_jpeg_extension(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot) return 0;

    // Compare extension case-insensitively
    if (strcasecmp(dot, ".jpg") == 0 || strcasecmp(dot, ".jpeg") == 0)
        return 1;

    return 0;
}

int load_images_from_folder(const char *path, image_t **dest, int max_images)
{
    DIR *dir;
    struct dirent *entry;
    int count = 0;

    if (!path || !dest || max_images <= 0)
    {
        log_error("invalid parameter");
        return 0;
    }

    dir = opendir(path);
    if (!dir) {
        log_error("opendir failed on %s",path);
        return 0;
    }

    int n=0;
    while ((entry = readdir(dir)) != NULL && count < max_images) {
        if (entry->d_type != DT_REG && entry->d_type != DT_UNKNOWN)
            continue;

        if (!has_jpeg_extension(entry->d_name))
            continue;

        n++;
        // Construct full file path
        char fullpath[4096];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", path, entry->d_name);

        image_t *img = load_jpeg(fullpath);
        if (img)
            dest[count++] = img;
        else
        {
            log_error("failed to load jpeg %s",fullpath);
        }
    }
    if (n==0)
    {
        log_error("no jpegs found in folder %s",path);
    }
    closedir(dir);
    return count;
}
