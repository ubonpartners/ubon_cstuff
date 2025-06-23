#ifndef __TRACKSET_H
#define __TRACKSET_H

#include <stdint.h>
#include <string.h>
#include <vector>
#include <string>
#include "detections.h"

typedef struct trackset trackset_t;

struct trackset_metadata {
    std::vector<std::string> classes;
    double                   frame_rate;
    int                      height;
    int                      width;
    std::string              original_video;
};

struct trackset
{
    trackset_metadata                metadata;
    std::vector<double>              frame_times;
    std::vector<detection_list_t *>  frame_detections;
};

trackset_t *trackset_load(const char *path);
void trackset_destroy(trackset_t *ts);

#endif
