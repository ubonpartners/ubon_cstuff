#include <time.h>
#include <string.h>
#include "ubon_cstuff_version.h"
#include "log.h"

bool is_jetson()
{
    #if (UBONCSTUFF_PLATFORM == 1)
    return true;
    #else
    return false;
    #endif
}