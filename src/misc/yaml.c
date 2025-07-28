#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <stdio.h>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <cstdarg>
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

void yaml_merge(YAML::Node baseNode, const YAML::Node& overrideNode)
{
    // If overrideNode is not a map, it completely replaces baseNode
    if (!overrideNode || overrideNode.Type() != YAML::NodeType::Map) {
        baseNode = overrideNode;
        return;
    }

    // If baseNode isn't already a map, make it one so we can merge into it
    if (!baseNode || baseNode.Type() != YAML::NodeType::Map) {
        baseNode = YAML::Node(YAML::NodeType::Map);
    }

    // Walk every key in overrideNode
    for (auto const& kv : overrideNode) {
        const auto& key = kv.first.as<std::string>();
        const auto& ov = kv.second;

        // If both sides are maps, merge recursively
        if (baseNode[key] &&
            baseNode[key].Type() == YAML::NodeType::Map &&
            ov.Type()       == YAML::NodeType::Map)
        {
            yaml_merge(baseNode[key], ov);
        }
        else {
            // Otherwise, override or insert
            baseNode[key] = ov;
        }
    }
}

void yaml_merge(YAML::Node baseNode, const char *yaml_or_file)
{
    YAML::Node node=yaml_load(yaml_or_file);
    yaml_merge(baseNode, node);
}

YAML::Node yaml_merge(const char *yaml_or_file_base, const char *yaml_or_file_to_merge)
{
    YAML::Node baseNode=yaml_load(yaml_or_file_base);
    YAML::Node node=yaml_load(yaml_or_file_to_merge);
    yaml_merge(baseNode, node);
    return baseNode;
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
        return node.as<int>();
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

const YAML::Node* yaml_traverse_path_count(const YAML::Node& base, int count, va_list args) {
    std::vector<YAML::Node> stack;
    stack.clear();
    stack.push_back(base);

    const YAML::Node* current = &base;
    for (int i = 0; i < count; ++i) {
        const char* key = va_arg(args, const char*);
        if (!current->IsMap())
            return nullptr;
        YAML::Node next = (*current)[key];
        if (!next)
            return nullptr;
        stack.push_back(next);
        current = &stack.back();
    }

    return current;
}

template<typename T>
T yaml_get_value_count(const YAML::Node& base, int count, T default_value, va_list args, T(*converter)(const YAML::Node&)) {
    const YAML::Node* final = yaml_traverse_path_count(base, count, args);
    if (final && final->IsDefined() && final->IsScalar()) {
        try {
            return converter(*final);
        } catch (const YAML::Exception&) {}
    }
    return default_value;
}

// Type-specific converters
int yaml_node_as_int(const YAML::Node& node) { return node.as<int>(); }
bool yaml_node_as_bool(const YAML::Node& node) { return node.as<bool>(); }
float yaml_node_as_float(const YAML::Node& node) { return node.as<float>(); }

// Entry-point wrappers with variadic unpacking
int yaml_get_int(const YAML::Node& base, int default_value, int count, ...) {
    va_list args;
    va_start(args, count);
    int val = yaml_get_value_count(base, count, default_value, args, yaml_node_as_int);
    va_end(args);
    return val;
}

bool yaml_get_bool(const YAML::Node& base, bool default_value, int count, ...) {
    va_list args;
    va_start(args, count);
    bool val = yaml_get_value_count(base, count, default_value, args, yaml_node_as_bool);
    va_end(args);
    return val;
}

float yaml_get_float(const YAML::Node& base, float default_value, int count, ...) {
    va_list args;
    va_start(args, count);
    float val = yaml_get_value_count(base, count, default_value, args, yaml_node_as_float);
    va_end(args);
    return val;
}