#include <math.h>
#include <stdio.h>
#include "image.h"

typedef double mat6x7[6][7];


/**
 * Solve 2D affine transform A x = b for 3 points
 * pts_src, pts_dst: [3][2]
 * M: output [2][3]
 */
// Compute affine transform via least-squares over 5 points
static void compute_affine5(const float src[5][2], const float dst[5][2], float M[6])
{
    // Build normal equations ATA * p = ATb for parameters p = [a00 a01 a02 a10 a11 a12]
    mat6x7 A = {0};
    // Accumulate
    for(int i=0;i<5;i++){
        double xs = src[i][0], ys = src[i][1];
        double xd = dst[i][0], yd = dst[i][1];
        // First equation row: [xs, ys, 1, 0, 0, 0] = xd
        double row1[6] = {xs, ys, 1, 0, 0, 0};
        // Second eq: [0,0,0, xs, ys, 1] = yd
        double row2[6] = {0, 0, 0, xs, ys, 1};
        // Update A (6x6) and b
        for(int r=0;r<6;r++){
            for(int c=0;c<6;c++) A[r][c] += row1[r]*row1[c] + row2[r]*row2[c];
            A[r][6]    += row1[r]*xd    + row2[r]*yd;
        }
    }
    // Solve 6x6 via Gauss elimination on augmented matrix A
    for(int i=0;i<6;i++){
        // pivot
        int piv=i;
        for(int j=i+1;j<6;j++) if(fabs(A[j][i])>fabs(A[piv][i])) piv=j;
        if(piv!=i) for(int k=i;k<=6;k++) std::swap(A[i][k], A[piv][k]);
        double diag = A[i][i]; if(fabs(diag)<1e-12) { diag=1e-12; }
        for(int k=i;k<=6;k++) A[i][k]/=diag;
        for(int j=0;j<6;j++) if(j!=i){ double factor=A[j][i]; for(int k=i;k<=6;k++) A[j][k]-=factor*A[i][k]; }
    }
    for(int i=0;i<6;i++) M[i] = (float)A[i][6];
}

// InsightFace canonical keypoints for 112x112 crop
static const float ref_pts[5][2] = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f}
};

void solve_affine_face_points(image_t **images, float *face_points, int n, int dest_w, int dest_h, float *M)
{
    for(int i=0;i<n;i++)
    {
        float ref_scale_x=dest_w/112.0;
        float ref_scale_y=dest_h/112.0;
        // build affine matrices
        for (int b = 0; b < n; ++b) {
            float src5[5][2];
            float dst5[5][2];
            if (face_points==0)
            {
                // this means images already aligned
                // we put in some dummy facepoints in the right place so that
                // the whole image gets used => still scaled to the right size
                for (int i = 0; i < 5; ++i) {
                    src5[i][0] = ref_pts[i][0] * images[b]->width / 112.0;
                    src5[i][1] = ref_pts[i][1] * images[b]->height / 112.0;
                }
            }
            else
            {
                for (int i = 0; i < 5; ++i) {
                    src5[i][0] = face_points[2*(5*b+i) + 0] * images[b]->width;
                    src5[i][1] = face_points[2*(5*b+i) + 1] * images[b]->height;
                }
            }
            for (int i = 0; i < 5; ++i) {
                dst5[i][0] = ref_pts[i][0]*ref_scale_x;
                dst5[i][1] = ref_pts[i][1]*ref_scale_y;
            }
            compute_affine5(dst5, src5, &M[6*b]);
        }
    }
}

static void solve_roi(image_t *img, roi_t roi, int dest_w, int dest_h, float *M)
{
    // compute the 6 affine warp parameters
    // that map the region roi[i] on image[i] to dest_w x dest_h

    // the cuda kernel maps dest pixel (x,y) to source pixel (sx,sy):
    // float sx = M[0] * x + M[1] * y + M[2];
    // float sy = M[3] * x + M[4] * y + M[5];

    float src_w=(float)(img->width);
    float src_h=(float)(img->height);
    float src_x0=roi.box[0]*src_w;
    float src_y0=roi.box[1]*src_h;
    float src_x1=roi.box[2]*src_w;
    float src_y1=roi.box[3]*src_h;

    M[0]=(src_x1-src_x0)/dest_w;
    M[1]=0;
    M[2]=src_x0;

    M[3]=0;
    M[4]=(src_y1-src_y0)/dest_h;
    M[5]=src_y0;
}

void solve_affine_points_roi(image_t **images, roi_t *rois, int n, int dest_w, int dest_h, float *M)
{
    for(int i=0;i<n;i++)
    {
        solve_roi(images[i], rois[i], dest_w, dest_h, M+6*i);
    }
}