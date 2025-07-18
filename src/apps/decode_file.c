#include <unistd.h>
#include "misc.h"
#include "display.h"
#include "profile.h"
#include "cuda_stuff.h"

static void process_image(void *context, image_t *img)
{
    double start_time= *((double *)context);
    if (img)
    {
        double target_time = img->time;
        while (profile_time() - start_time < target_time)
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
