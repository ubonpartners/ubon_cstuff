#ifndef __WEBCAM_H
#define __WEBCAM_H

#include "image.h"

typedef struct webcam webcam_t;

webcam_t *webcam_create(const char *device, int width, int height);
void webcam_destroy(webcam_t *w);
image_t *webcam_capture(webcam_t *w);

#endif
