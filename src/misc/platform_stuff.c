#include <time.h>
#include <string.h>
#include "ubon_cstuff_version.h"
#include "log.h"
#include "platform_stuff.h"
#include "memory_stuff.h"
#include "yaml_stuff.h"

bool platform_is_jetson()
{
    #if (UBONCSTUFF_PLATFORM == 1)
    return true;
    #else
    return false;
    #endif
}

const char *platform_get_stats()
{
    YAML::Node node=allocation_tracker_stats_node();
    return yaml_to_cstring(node);
}