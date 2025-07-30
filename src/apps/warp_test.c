#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <sys/stat.h>
#include <string.h>

#include "pcap_stuff.h"
#include "rtp_receiver.h"
#include "h26x_assembler.h"
#include "simple_decoder.h"
#include "pcap_decoder.h"
#include "display.h"
#include "cuda_stuff.h"
#include "cuda_kernels.h"
#include "image.h"
#include "jpeg.h"
#include "infer_aux.h"


int main(int argc, char *argv[])
{
    init_cuda_stuff();
    image_init();

    if (1)
    {
        image_t *img0=load_jpeg("/mldata/image/arrest2.jpg");
        image_t *img=image_convert(img0, IMAGE_FORMAT_YUV420_DEVICE);
        image_t *img_rgb=image_create(112, 112, IMAGE_FORMAT_RGB24_DEVICE);

        float a=0.1;
        float s=1;
        while(1)
        {
            float matrix[6]={s*sin(a), s*cos(a), 400.0, s*cos(a), -s*sin(a), 400.0};
            a=a+0.05;
            s=2.0+sin(a*5);//s*1.01;
            image_t *images[12];
            images[0]=img;
            cuda_warp_yuv420_to_planar_float(
                (const image_t**)&images[0],
                (void*)img_rgb->rgb,
                1,
                img_rgb->width, img_rgb->height,
                (const float *)&matrix[0],
                true, false,
                img_rgb->stream);

            display_image("test", img_rgb);

            usleep(30*1000);
        }
    }

    infer_aux_t *inf=infer_aux_create("/mldata/facerec/r18_glint360k-dyn.trt", 0);


}
