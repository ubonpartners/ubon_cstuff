#ifndef __PROFILE_H
#define __PROFILE_H

#include <stdio.h>
#include <time.h>

static inline double profile_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);  // or CLOCK_MONOTONIC_RAW
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

#endif
