#include <stdint.h>
#include "string.h"
#include "assert.h"
#include "memory_stuff.h"
#include "embedding.h"
#include "detections.h"
#include "infer.h"
#include "log.h"
#include "misc.h"
#include <math.h>


const char *blur_str[] = {"hazy", "blurry", "clear"};
const char *occ_str[] = {"obstructed", "unobstructed"};
const char *pose_str[]= {"profile", "slight angle", "frontal"};
const char *exp_str[]= {"exaggerated expression", "typical expression"};
const char *ill_str[]= {"extreme lighting", "normal lighting"};
const char *qual_str[]= {"bad", "poor", "fair", "good", "perfect"};

static void softmax_inplace(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val)
            max_val = input[i];
    }

    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }

    // Normalize in-place
    for (int i = 0; i < size; ++i) {
        input[i] /= sum;
    }
}

static void reduce_along_axes(float* input, float* out_axis0, float* out_axis1, float* out_axis2,
                       float* out_axis3, float* out_axis4, float* out_axis5) {
    int idx = 0;
    for (int i0 = 0; i0 < 3; ++i0) {
        for (int i1 = 0; i1 < 2; ++i1) {
            for (int i2 = 0; i2 < 3; ++i2) {
                for (int i3 = 0; i3 < 2; ++i3) {
                    for (int i4 = 0; i4 < 2; ++i4) {
                        for (int i5 = 0; i5 < 5; ++i5) {
                            float val = input[idx++];

                            out_axis0[i0] += val;
                            out_axis1[i1] += val;
                            out_axis2[i2] += val;
                            out_axis3[i3] += val;
                            out_axis4[i4] += val;
                            out_axis5[i5] += val;
                        }
                    }
                }
            }
        }
    }
}

static int maxidx(float *p, int n)
{
    float v=0;
    int ret=0;
    for(int i=0;i<n;i++)
    {
        if (p[i]>v)
        {
            v=p[i];
            ret=i;
        }
    }
    return ret;
}

float fiqa_embedding_quality(embedding_t *e)
{
    float *p=embedding_get_data(e);
    assert(embedding_get_size(e)==360);
    //for(int i=0;i<360;i++) printf("%3d) %f\n",i,p[i]);
    softmax_inplace(p, 360);
    // reduce just over last dimension
    float quality[5]={0,0,0,0,0};
    for (int i=0;i<360;i+=5)
    {
        quality[0]+=p[i+0];
        quality[1]+=p[i+1];
        quality[2]+=p[i+2];
        quality[3]+=p[i+3];
        quality[4]+=p[i+4];
    }
    const float qual_dot[5]={0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float q=0;
    for(int i=0;i<5;i++) q+=quality[i]*qual_dot[i];
    return q;
}

void fiqa_embedding_show(embedding_t *e)
{
    float *p=embedding_get_data(e);
    assert(embedding_get_size(e)==360);
    //for(int i=0;i<360;i++) printf("%3d) %f\n",i,p[i]);
    softmax_inplace(p, 360);

    float blur[3]={0,0,0};
    float occ[2]={0,0};
    float pose[3]={0,0,0};
    float exp[2]={0,0};
    float ill[2]={0,0};
    float qual[5]={0,0,0,0,0};

    reduce_along_axes(p, blur, occ, pose, exp, ill, qual);

    float qual_dot[5]={0.1f, 0.3f, 0.5f, 0.7f, 0.9f};

    float q=0;
    for(int i=0;i<5;i++) q+=qual[i]*qual_dot[i];

    /*printf("%f %f\n",exp[0],exp[1]);

    printf("Quality %f : %8s %8s %8s %8s %8s %8s\n",q,
        blur_str[maxidx(blur,3)],
        occ_str[maxidx(occ, 2)],
        pose_str[maxidx(pose, 3)],
        exp_str[maxidx(exp, 2)],
        ill_str[maxidx(ill, 2)],
        qual_str[maxidx(qual, 5)]
    );*/
}

float cevo_bestface_score(detection_t *det) {
    const float LANDMARK_CONF_THRESH=0.8f;
    const int NUM_LANDMARKS = 5;
    const float CONFIDENCE_WEIGHT = 0.3f;
    const float FRONTALITY_WEIGHT = 0.4f;
    const float SYMMETRY_WEIGHT = 0.3f;

    // 1. VALIDATE INPUT
    if (det->num_face_points!=NUM_LANDMARKS) {
        return 0.0f;
    }

    // 2. CHECK LANDMARK CONFIDENCE (with averaging)
    float avg_confidence = 0.0f;
    for(int i = 0; i < NUM_LANDMARKS; i++) {
        float conf=det->face_points[i].conf;
        if (conf < LANDMARK_CONF_THRESH) {
            return 0.0f; // Hard threshold for minimum quality
        }
        avg_confidence += conf;
    }
    avg_confidence /= NUM_LANDMARKS;

    // 3. CALCULATE NORMALIZED COORDINATES
    float width = det->subbox_x1- det->subbox_x0;
    float height = det->subbox_y1- det->subbox_y0;
    if (width <= 0 || height <= 0) return 0.0f;

    // Nose position (normalized)
    float nose_x = (det->face_points[2].x-det->subbox_x0) / width;
    float nose_y = (det->face_points[2].y-det->subbox_y0) / height;

    // 4. IMPROVED FRONTALITY CALCULATION
    // Calculate face center without nose
    float face_center_x = 0.0f, face_center_y = 0.0f;
    int valid_points = 0;

    for(int i = 0; i < NUM_LANDMARKS; i++) {
        if (i == 2) continue; // Skip nose (assuming index 2)

        face_center_x += (det->face_points[i].x - det->subbox_x0) / width;
        face_center_y += (det->face_points[i].y - det->subbox_y0) / height;
        valid_points++;
    }

    if (valid_points > 0) {
        face_center_x /= valid_points;
        face_center_y /= valid_points;
    }

    // 5. MULTIPLE QUALITY METRICS

    // A. Frontality Score (how centered is the nose)
    float nose_deviation = sqrtf(powf(nose_x - 0.5f, 2) + powf(nose_y - 0.5f, 2));
    float frontality_score = fmaxf(0.0f, 1.0f - nose_deviation * 2.0f);

    // B. Symmetry Score (left-right eye balance)
    float left_eye_x = (det->face_points[0].x - det->subbox_x0) / width;
    float right_eye_x = (det->face_points[1].x  - det->subbox_x0) / width;
    float eye_symmetry = 1.0f - fabsf((left_eye_x + right_eye_x) / 2.0f - 0.5f) * 2.0f;

    // C. Combined Score
    float quality_score = (avg_confidence * CONFIDENCE_WEIGHT) +
                            (frontality_score * FRONTALITY_WEIGHT) +
                            (fmaxf(0.0f, eye_symmetry) * SYMMETRY_WEIGHT);

    return fminf(1.0f, fmaxf(0.0f, quality_score));
}
