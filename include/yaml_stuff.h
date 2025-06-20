#ifndef __YAML_H
#define __YAML_H

#include <yaml-cpp/yaml.h>

YAML::Node yaml_load(const char *yaml_or_file);
const char *yaml_to_cstring(YAML::Node node);
float yaml_get_float_value(YAML::Node node, float dv);
int yaml_get_int_value(YAML::Node node, int dv);

#endif
