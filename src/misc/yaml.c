#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <stdio.h>
#include <vector>
#include <optional>
#include <yaml-cpp/yaml.h>
#include <cstdarg>
#include "yaml_stuff.h"
#include "log.h"
#include "platform_stuff.h"

YAML::Node yaml_load(const char* input, bool set_platform)
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

    if (set_platform)
    {
        bool is_jetson=platform_is_jetson();
        if (is_jetson)
        {
            filterNode(ret, "(platform:x86)");
            renameKeys(ret, "(platform:jetson)");
        }
        else
        {
            filterNode(ret, "(platform:jetson)");
            renameKeys(ret, "(platform:x86)");
        }
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
    if (yaml_or_file==0) return;
    YAML::Node node=yaml_load(yaml_or_file);
    yaml_merge(baseNode, node);
}

YAML::Node yaml_merge(const char *yaml_or_file_base, const char *yaml_or_file_to_merge)
{
    YAML::Node baseNode=yaml_load(yaml_or_file_base);
    if (yaml_or_file_to_merge!=0)
    {
        YAML::Node node=yaml_load(yaml_or_file_to_merge);
        yaml_merge(baseNode, node);
    }
    return baseNode;
}

const char *yaml_merge_string(const char *yaml_or_file_base, const char *yaml_or_file_to_merge)
{
    YAML::Node node=yaml_merge(yaml_or_file_base, yaml_or_file_to_merge);
    return yaml_to_cstring(node);
}

const char *yaml_to_cstring(YAML::Node node)
{
    YAML::Emitter out;
    out.SetIndent(4);
    out.SetMapFormat(YAML::Auto);
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

std::optional<YAML::Node> yaml_traverse_path_count(const YAML::Node& base, int count, va_list args)
{
    YAML::Node current = base;  // copy the handle (cheap)
    for (int i = 0; i < count; ++i) {
        const char* key = va_arg(args, const char*);
        if (!current || !current.IsMap())
            return std::nullopt;

        YAML::Node next = current[key];  // safe handle
        if (!next || !next.IsDefined())
            return std::nullopt;

        current = next;  // keep walking by value
    }
    return current;  // returned by value (no dangling)
}

template<typename T>
T yaml_get_value_count(const YAML::Node& base,
                       int count,
                       T default_value,
                       va_list args,
                       T(*converter)(const YAML::Node&))
{
    auto final = yaml_traverse_path_count(base, count, args);
    if (final && final->IsDefined() && final->IsScalar()) {
        try { return converter(*final); }
        catch (const YAML::Exception&) {}
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

void filterNode(YAML::Node& node, const std::string& substr) {
    switch (node.Type()) {
        case YAML::NodeType::Map: {
            // First, process children recursively
            for (auto it = node.begin(); it != node.end(); ++it) {
                const std::string key = it->first.as<std::string>();
                if (key.find(substr) == std::string::npos) {
                    filterNode(it->second, substr);
                }
            }
            // Collect keys to remove
            std::vector<std::string> to_remove;
            for (auto it = node.begin(); it != node.end(); ++it) {
                const std::string key = it->first.as<std::string>();
                if (key.find(substr) != std::string::npos) {
                    to_remove.push_back(key);
                } else {
                    // Also prune empty child maps/sequences
                    YAML::Node child = it->second;
                    if ((child.IsMap()  && child.size() == 0) ||
                        (child.IsSequence() && child.size() == 0)) {
                        to_remove.push_back(key);
                    }
                }
            }
            // Actually remove them
            for (const auto& k : to_remove) {
                node.remove(k);
            }
            break;
        }
        case YAML::NodeType::Sequence: {
            // Process each element recursively, collect survivors
            std::vector<YAML::Node> survivors;
            for (auto elem : node) {
                filterNode(elem, substr);
                if (!( (elem.IsMap()      && elem.size() == 0) ||
                       (elem.IsSequence() && elem.size() == 0) )) {
                    survivors.push_back(elem);
                }
            }
            // Overwrite sequence with survivors
            node = YAML::Node(YAML::NodeType::Sequence);
            for (auto& e : survivors) {
                node.push_back(e);
            }
            break;
        }
        default:
            // Scalars etc. – nothing to do
            break;
    }
}

// Recursively rename keys in-place: any map-key containing 'substr' will
// have all occurrences of 'substr' removed from its name.
// If renaming would collide with an existing key whose value is a sequence
// and the new value is also a sequence, merge the two sequences.
void renameKeys(YAML::Node& node, const std::string& substr) {
    if (node.IsMap()) {
        // 1) Recurse into all child values
        for (auto it = node.begin(); it != node.end(); ++it) {
            renameKeys(it->second, substr);
        }

        // 2) Collect keys to rename
        struct Op { std::string oldKey, newKey; YAML::Node value; };
        std::vector<Op> ops;
        for (auto it = node.begin(); it != node.end(); ++it) {
            const std::string key = it->first.as<std::string>();
            if (key.find(substr) != std::string::npos) {
                // strip all occurrences of substr
                std::string newKey = key;
                size_t pos;
                while ((pos = newKey.find(substr)) != std::string::npos) {
                    newKey.erase(pos, substr.size());
                }
                ops.push_back({ key, newKey, it->second });
            }
        }

        // 3) Apply renames (remove oldKey, then merge or overwrite)
        for (auto& op : ops) {
            node.remove(op.oldKey);
            YAML::Node existing = node[op.newKey];
            if (existing && existing.IsSequence() && op.value.IsSequence()) {
                // merge sequences
                for (std::size_t j = 0; j < op.value.size(); ++j) {
                    existing.push_back(op.value[j]);
                }
            } else {
                node[op.newKey] = op.value;
            }
        }
    }
    else if (node.IsSequence()) {
        // Copy each element out (so we get a real YAML::Node for recursion),
        // recurse on it, then write it back.
        for (std::size_t i = 0; i < node.size(); ++i) {
            YAML::Node elem = node[i];         // copy the element
            renameKeys(elem, substr);          // recurse on the copy
            node[i] = elem;                    // write it back in place
        }
    }
    // scalars/null: nothing to do
}