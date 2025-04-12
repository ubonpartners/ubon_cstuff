#ifndef __DATASET_H
#define __DATASET_H

typedef struct dataset dataset_t;

#include "image.h"
#include "detections.h"

dataset_t *dataset_create(const char *path_to_images);
void dataset_destroy(dataset_t *ds);
int dataset_get_num(dataset_t *ds);
const char *dataset_get_image_path(dataset_t *ds, int index);
const char *dataset_get_label_path(dataset_t *ds, int index);
image_t *dataset_get_image(dataset_t *ds, int index);
detections_t *dataset_get_gts(dataset_t *ds, int index);

#endif