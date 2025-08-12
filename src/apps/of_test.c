

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
#include "nvof.h"
#include "image_draw.h"
#include "file_decoder.h"

static nvof_t *nvof=0;

/*typedef struct flow_vector
{
    int16_t                         flowx;
    int16_t                         flowy;
} flow_vector_t;

typedef struct nvof_result
{
    int grid_w, grid_h;
    uint8_t *costs;
    flow_vector_t *flow;
} nvof_results_t;
*/
static void process_image(void *context, image_t *img)
{
    nvof_results_t *r=nvof_execute(nvof, img);

    image_t *s=image_convert(img, IMAGE_FORMAT_RGB24_HOST);
    image_sync(s);
    assert(s!=0);
    assert(s->format==IMAGE_FORMAT_RGB24_HOST);

    image_draw_box(s, 0.1, 0.1, 0.9, 0.9, 0xffff0000);

    display_image("video", s);

    if (r->flow)
    {
        for(int y=0;y<r->grid_h;y++)
        {
            for(int x=0;x<r->grid_w;x++)
            {
                float cx=(x+0.5)/r->grid_w;
                float cy=(y+0.5)/r->grid_h;
                int dx=r->flow[x+y*r->grid_w].flowx;
                int dy=r->flow[x+y*r->grid_w].flowy;
                float dxf=dx/(4.0f*32.0f*r->grid_w);
                float dyf=dy/(4.0f*32.0f*r->grid_w);
                image_draw_line(s, cx,cy, cx+dxf,cy+dyf,0xffff0000);
            }
        }
    }

    image_destroy(s);
}

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();

    nvof=nvof_create(0, 320, 320);

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