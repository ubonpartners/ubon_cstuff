#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <stdio.h>
#include "yaml_stuff.h"
#include "log.h"

YAML::Node yaml_load(const char* input)
{
    std::string str_input(input);
    YAML::Node ret;
    // Check for .yaml or .yml extension (case-sensitive)
    auto has_yaml_extension = [](const std::string& path) {
        return path.size() >= 5 && (
            path.compare(path.size() - 5, 5, ".yaml") == 0 ||
            path.compare(path.size() - 5, 5, ".json") == 0 ||
            (path.size() >= 4 && path.compare(path.size() - 4, 4, ".yml") == 0)
        );
    };

    try {
        if (has_yaml_extension(str_input)) {
            // Try to open the file first
            std::ifstream file(str_input);
            if (!file.good()) {
                throw std::runtime_error("YAML file does not exist or cannot be opened: " + str_input);
            }
            ret=YAML::LoadFile(str_input);
        } else {
            // Treat it as a YAML string
            ret=YAML::Load(str_input);
        }
    } catch (const YAML::ParserException& e) {
        std::string error_msg = std::string("YAML parse error: ") + e.what();
        const char* c_error_msg = error_msg.c_str();
        log_fatal("YAML parse error: input='%s'; error is '%s'",input,c_error_msg);
        assert(0);
    }
    return ret;
}

const char *yaml_to_cstring(YAML::Node node)
{
    YAML::Emitter out;
    out << node;

    // Get the YAML as a C-string
    const char* yaml_str = out.c_str();
    return strdup(yaml_str);
}

float yaml_get_float_value(YAML::Node node, float dv)
{
    if (node && node.IsDefined()) {
        return node.as<float>();
    }
    return dv;
}

int yaml_get_int_value(YAML::Node node, int dv)
{
    if (node && node.IsDefined()) {
        return node.as<float>();
    }
    return dv;
}

bool yaml_get_bool_value(YAML::Node node, bool dv)
{
    if (node && node.IsDefined()) {
        return node.as<bool>();
    }
    return dv;
}
