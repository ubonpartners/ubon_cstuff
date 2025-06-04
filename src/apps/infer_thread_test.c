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
#include "infer_thread.h"
#include "nvof.h"
#include "misc.h"

static infer_thread_t *infer_thread=0;

static void frame_callback(void *context, image_t *img)
{
    // show inference working by doing two lots of inference with different ROIs
    roi_t roi_a, roi_b;
    roi_a.box[0]=0.25;
    roi_a.box[1]=0.25;
    roi_a.box[2]=0.75;
    roi_a.box[3]=0.75;

    roi_b.box[0]=0.1;
    roi_b.box[1]=0.1;
    roi_b.box[2]=0.2;
    roi_b.box[3]=0.8;

    infer_thread_result_handle_t *ha=infer_thread_infer_async(infer_thread, img, roi_a);
    infer_thread_result_handle_t *hb=infer_thread_infer_async(infer_thread, img, roi_b);
    infer_thread_result_data_t da={};
    infer_thread_result_data_t db={};
    infer_thread_wait_result(ha, &da);
    infer_thread_wait_result(hb, &db);

    detections_t *all_dets=detections_join(da.dets, db.dets);
    image_t *out_frame=draw_detections(all_dets, img);

    // uncommend to print detections
    //show_detections(all_dets);

    destroy_detections(all_dets);
    destroy_detections(da.dets);
    destroy_detections(db.dets);

    display_image("video", out_frame);
    destroy_image(out_frame);
}

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();

    infer_config_t config={};
    config.det_thr=0.4;
    config.set_det_thr=true;
    config.nms_thr=0.5;
    config.set_nms_thr=true;

    infer_thread=infer_thread_start("/mldata/weights/trt/yolo11l-dpa-131224.trt",
                                    "/mldata/config/train/train_attr.yaml", &config);

    model_description_t *md=infer_thread_get_model_description(infer_thread);
    infer_print_model_description(md);

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
            frame_callback(0, i);
            destroy_image(i);
        }
    }


    return 0;
}
