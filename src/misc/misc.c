#include <time.h>
#include "ubon_cstuff_version.h"

double time_now_sec() 
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

const char *ubon_cstuff_get_version(void)
{
    const char *version = (const char*)UBON_CSTUFF_VERSION;

    return version;
}
