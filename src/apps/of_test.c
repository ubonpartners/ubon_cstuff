
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cuda_stuff.h"
#include "simple_decoder.h"
#include "display.h"
#include "webcam.h"
#include "track.h"
#include "misc.h"
#include "profile.h"
#include "trackset.h"


static void process_image(void *context, image_t *img)
{
    display_image("video", img);
}

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();

    if (argc>1)
    {
        decode_file(argv[1], 0, process_image, 0);
    }
    else
    {
        webcam_t *w=webcam_create("/dev/video0", 1280, 720);

        while(1)
        {
            image_t *i=webcam_capture(w);
            process_image(0, i);
            image_destroy(i);
        }
    }

    return 0;
}