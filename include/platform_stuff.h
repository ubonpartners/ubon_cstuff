#ifndef __PLATFORM_STUFF_H
#define __PLATFORM_STUFF_H

#include <yaml-cpp/yaml.h>

bool platform_is_jetson();

const char *platform_get_stats(); // returns YAML string

#endif