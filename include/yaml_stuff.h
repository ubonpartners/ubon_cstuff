#ifndef __YAML_H
#define __YAML_H

#include <yaml-cpp/yaml.h>

YAML::Node yaml_load(const char *yaml_or_file, bool set_platform=true);
void yaml_merge(YAML::Node baseNode, const YAML::Node& overrideNode);
void yaml_merge(YAML::Node baseNode, const char *yaml_or_file);
YAML::Node yaml_merge(const char *yaml_or_file_base, const char *yaml_or_file_to_merge);
const char *yaml_merge_string(const char *yaml_or_file_base, const char *yaml_or_file_to_merge);
void renameKeys(YAML::Node& node, const std::string& substr);
void filterNode(YAML::Node& node, const std::string& substr);
const char *yaml_to_cstring(YAML::Node node);
float yaml_get_float_value(YAML::Node node, float dv);
int yaml_get_int_value(YAML::Node node, int dv);
bool yaml_get_bool_value(YAML::Node node, bool  dv);

int yaml_get_int(const YAML::Node& base, int default_value, int count,  ...);
bool yaml_get_bool(const YAML::Node& base, bool default_value, int count,  ...);
float yaml_get_float(const YAML::Node& base, float default_value, int count,  ...);

#endif
