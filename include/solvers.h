#ifndef __SOLVERS_H
#define __SOLVERS_H

#include "image.h"

void solve_affine_face_points(image_t **images, float *face_points, int n, int dest_w, int dest_h, float *M);

#endif