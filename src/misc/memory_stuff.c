#include "misc.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <cstring>      // strdup
#include <string>
#include <vector>
#include <mutex>
#include <cstdio>       // snprintf

// internal registry entry
struct TrackerEntry {
    allocation_tracker_t* tracker;
    std::string          name;
};

static std::vector<TrackerEntry> registry;
static std::mutex               registry_mutex;

void allocation_tracker_register(allocation_tracker_t *t, const char *name)
{
    std::lock_guard<std::mutex> lock(registry_mutex);
    registry.push_back(TrackerEntry{ t, name });
}

char *allocation_tracker_stats(void)
{
    std::lock_guard<std::mutex> lock(registry_mutex);

    std::string report;
    report.reserve(1024);
    report += "Allocation Tracker Stats:\n";

    char linebuf[256];
    for (auto &e : registry) {
        int n = std::snprintf(
            linebuf, sizeof(linebuf),
            "  %30s: allocs=%8llu, frees=%8llu, total_alloc=%14llu, total_free=%14llu outstanding=%6llu\n",
            e.name.c_str(),
            (unsigned long long)e.tracker->num_allocs,
            (unsigned long long)e.tracker->num_frees,
            (unsigned long long)e.tracker->total_alloc,
            (unsigned long long)e.tracker->total_free,
            (unsigned long long)(e.tracker->num_allocs-e.tracker->num_frees)
        );
        if (n > 0) {
            report.append(linebuf, static_cast<size_t>(n));
        }
    }

    // Return a strdup'ed C string; caller must free().
    return strdup(report.c_str());
}