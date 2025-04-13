#ifndef __VPI_H
#define __VPI_H

#include <stdint.h>

typedef struct vpi_of vpi_of_t;

#include "image.h"

vpi_of_t *vpi_of_create(void *context, int w, int h);
void vpi_of_destroy(vpi_of_t *v);
void vpi_of_execute(vpi_of_t *v, image_t *img);

#endif
