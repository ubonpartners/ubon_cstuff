#include "log.h"
#include "cuda_stuff.h"
#include "nvof.h"
#include "display.h"
#include <stdio.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <mutex>

#if (UBONCSTUFF_PLATFORM == 1) // Orin Nano
struct nvof
{
    int max_width;
    int max_height;
    int width;
    int height;
    int gridsize;
    int outW;
    int outH;

    nvof_results_t results;
};


nvof_results_t *nvof_execute(nvof_t *v, image_t *img_in)
{
    (void)v; (void)img_in;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return NULL;
}

nvof_t *nvof_create(void *context, int max_width, int max_height) 
{
    (void)context; (void)max_width; (void)max_height;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return NULL;
}

void nvof_destroy(nvof_t *n) 
{
    (void)n;
    log_error("%s:%d This feature is not yet implemented", __func__, __LINE__);

    return;
}
#endif //(UBONCSTUFF_PLATFORM == 1)

