#include <unistd.h>
#include "misc.h"
#include "display.h"
#include "profile.h"
#include "cuda_stuff.h"
#include "file_decoder.h"

static void process_image(void *context, image_t *img)
{
    if (img)
    {
        static double first_frame=true;
        static double start_profile_time;
        static double start_frame_time;
        if (first_frame)
        {
            start_profile_time=profile_time();
            start_frame_time = img->meta.time;
            first_frame=false;
        }
        printf("process image: image time %f\n",img->meta.time);
        double target_time = img->meta.time-start_frame_time;
        while (profile_time() - start_profile_time < target_time)
        {
            usleep(1000); // wait until the target time is reached
        }
        display_image("video", img);
    }
}

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();
    double start_time = profile_time();
    if (argc>1)
    {
        decode_file(argv[1], &start_time, process_image, 0);
    }
    return 1;
}
