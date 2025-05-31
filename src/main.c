#include "simple_decoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cuda_stuff.h"
#include "display.h"
#include "webcam.h"
#include "infer.h"
#include "jpeg.h"
#include "dataset.h"
#include "nvof.h"
#include "misc.h"
#include "c_tests.h"

static infer_t *inf=0;
static nvof_t *v=0;

static void frame_callback(void *context, image_t *img)
{
    display_t *d=(display_t *)context;
    detections_t *dets=infer(inf, img);
    image_t *blurred=image_blur(img);
    image_t *out_frame_rgb=draw_detections(dets, blurred);
    destroy_image(blurred);

    nvof_execute(v, img);

    display_image("video", out_frame_rgb);
    destroy_image(out_frame_rgb);
    destroy_detections(dets);
}

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();

    if (argc>1 && strcmp(argv[1], "--test")==0)
    {
        return run_all_c_tests();
    }

    /*dataset_t *ds=dataset_create("/mldata/coco-face/val/images");
    int num=dataset_get_num(ds);
    for(int i=0;i<num;i++)
    {
        image_t *im=dataset_get_image(ds,i);
        detections_t *gts=dataset_get_gts(ds, i);
        //image_t *im2=draw_detections(gts,im);
        //display_image("this",im2);
        //usleep(1000000*10);
        destroy_image(im);
        //destroy_image(im2);
        destroy_detections(gts);
    }*/

    inf=infer_create("/mldata/weights/trt/yolo11l-dpa-131224.trt", "/mldata/config/train/train_attr.yaml");
    //inf=infer_create("/mldata/weights/trt/yolo11l-dpa-131224_dyn.trt", "/mldata/config/train/train_attr.yaml");
    v=nvof_create(0,320,320);

    if (argc>1)
    {
        FILE *input = fopen(argv[1], "rb");
        if (!input)
        {
            printf("Failed to open input file %s", argv[1]);
            return -1;
        }

        simple_decoder_t *decoder = simple_decoder_create(0, frame_callback, SIMPLE_DECODER_CODEC_H264);

        uint8_t buffer[4096];
        size_t bytes_read;
        while ((bytes_read = fread(buffer, 1, sizeof(buffer), input)) > 0)
        {
            simple_decoder_decode(decoder, buffer, bytes_read);
        }

        simple_decoder_destroy(decoder);
        fclose(input);
    }
    else
    {
        webcam_t *w=webcam_create("/dev/video0", 1280, 720);

        while(1)
        {
            image_t *i=webcam_capture(w);
            detections_t *dets=infer(inf, i);
            image_t *out=draw_detections(dets, i);
            display_image("infer", out);
            destroy_image(out);
            destroy_detections(dets);
            destroy_image(i);
        }
    }


    return 0;
}
