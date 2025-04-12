#ifndef __DISPLAY_H
#define __DISPLAY_H

typedef struct display display_t;

#include "image.h"

display_t *display_create(const char *title);
void display_destroy(display_t *d);
void display_image(display_t *d, image_t *img);
void display_image(const char *txt, image_t *img);

#endif