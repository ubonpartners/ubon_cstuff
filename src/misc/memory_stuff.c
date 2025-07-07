#include "misc.h"
#include "memory_stuff.h"
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <cstring>      // strdup
#include <string>
#include <vector>
#include <mutex>
#include <cstdio>       // snprintf
#include <unordered_map>
#include <mutex>
#include <new>   // for std::nothrow
#include "log.h"

#define BLOCKS_PER_BIG 64
#define BA_CANARY           0xFEEDBEEF1234ULL
#define BIG_CANARY          0xBADC0FFEE0DDF00DULL
#define SUB_CANARY          0xC0FFEEBABECAFEULL
#define SUB_CANARY_DEAD     0xDEADBADDEADBADLL

typedef struct block_allocator block_allocator_t;

typedef struct big_block {
    uint64_t canary;
    uint64_t free_mask;
    size_t block_size;
    struct big_block *next_free, *prev_free;
    struct big_block *next_all, *prev_all;
    block_allocator_t *allocator;
} big_block_t;

typedef struct block_header {
    block_allocator_t *block_allocator;
    big_block_t *parent;
    void (*free_callback)(void *context, void *block);
    void *callback_context;
    unsigned int ref_count;
    unsigned int idx;
    unsigned int block_size;
    unsigned int dummy;
    uint64_t canary;
} block_header_t;

struct block_allocator {
    size_t block_size;
    big_block_t *free_list_head;
    big_block_t *all_blocks;
    pthread_mutex_t lock;
    size_t total_big_blocks;
    size_t free_big_blocks;
    size_t allocated_blocks;
    size_t high_watermark_big_blocks;
    size_t total_variable_blocks;
    size_t total_variable_memory;
    size_t total_variable_memory_hwm;
    uint64_t canary;
};

// internal registry entry
struct TrackerEntry {
    allocation_tracker_t* tracker;
    block_allocator_t *block_allocator;
    std::string          name;
};

static std::vector<TrackerEntry> registry;
static std::mutex               registry_mutex;

struct allocation_table
{
    std::unordered_map<void*, size_t> map;
    std::mutex                       mutex;
};

allocation_table_t *allocation_table_create()
{
    return reinterpret_cast<allocation_table_t*>(new (std::nothrow) allocation_table());
}

void allocation_table_insert(allocation_table_t *table, void *ptr, size_t size)
{
    if (!table || !ptr) return;
    auto* t = reinterpret_cast<allocation_table*>(table);
    std::lock_guard<std::mutex> lk(t->mutex);
    auto it = t->map.find(ptr);
    //assert(it == t->map.end());
    if (it != t->map.end())
    {
        log_error("allocation_table_add: double-insert pointer %p already in table (sizes %ld, %ld)",
            ptr, it->second, size);
        return;
    }
    t->map[ptr] = size;
}

size_t allocation_table_remove(allocation_table_t *table, void *ptr)
{
    if (!table || !ptr) return 0;
    auto* t = reinterpret_cast<allocation_table*>(table);
    std::lock_guard<std::mutex> lk(t->mutex);
    auto it = t->map.find(ptr);
    //assert(it != t->map.end());
    if (it == t->map.end())
    {
        log_error("allocation_table_remove: pointer %p not found in table", ptr);
        return 0;
    }
    size_t sz = it->second;
    t->map.erase(it);
    return sz;
}

void allocation_table_check_ptr(allocation_table_t *table, void *ptr)
{
    if (!table || !ptr) return;
    auto* t = reinterpret_cast<allocation_table*>(table);
    std::lock_guard<std::mutex> lk(t->mutex);
    auto it = t->map.find(ptr);
    //assert(it != t->map.end());
    if (it == t->map.end()) log_error("allocation_table_check pointer %p not found in table", ptr);
}

void allocation_tracker_register(allocation_tracker_t *t, const char *name, bool use_table)
{
    if (use_table) t->allocation_table=allocation_table_create();
    std::lock_guard<std::mutex> lock(registry_mutex);
    registry.push_back(TrackerEntry{ t, 0, name });
}

void block_allocator_register(block_allocator_t *t, const char *name)
{
    std::lock_guard<std::mutex> lock(registry_mutex);
    registry.push_back(TrackerEntry{ 0, t, name });
}

void allocation_tracker_reset()
{
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (auto &e : registry) {
        if (e.tracker!=0)
        {
            e.tracker->num_hwm=0;
            e.tracker->total_hwm=0;
        }
        if (e.block_allocator!=0)
        {
            e.block_allocator->high_watermark_big_blocks=0;
            e.block_allocator->total_variable_memory_hwm=0;
        }
    }
}

double allocation_tracker_get_mem_HWM(const char *tracker_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (auto &e : registry) {
        if (strcmp(tracker_name, e.name.c_str())==0)
        {
            if (e.tracker!=0) return (double)e.tracker->total_hwm;
        }
    }
    assert(0);
}

double allocation_tracker_get_mem_outstanding(const char *tracker_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (auto &e : registry) {
        if (strcmp(tracker_name, e.name.c_str())==0)
        {
            if (e.tracker!=0) return ((double)(e.tracker->total_alloc-e.tracker->total_free));
        }
    }
    assert(0);
}

char *allocation_tracker_stats(void)
{
    std::lock_guard<std::mutex> lock(registry_mutex);

    std::string report;
    report.reserve(1024);
    report += "Allocation Tracker Stats:\n";

    auto format_size = [](double bytes) -> std::string {
        double mb = bytes / 1e6;
        char buf[32];
        if (mb >= 1000.0)
            std::snprintf(buf, sizeof(buf), "%.2fGB", mb / 1000.0);
        else
            std::snprintf(buf, sizeof(buf), "%.1fMB", mb);
        return std::string(buf);
    };

    char linebuf[256];
    for (auto &e : registry) {
        if (e.tracker!=0)
        {
            int n = std::snprintf(
                linebuf, sizeof(linebuf),
                "  %30s: num: alloc=%8llu free=%8llu hwm=%8llu outs=%6llu, total: alloc=%8s hwm=%8s outs=%8s\n",
                e.name.c_str(),
                (unsigned long long)e.tracker->num_allocs,
                (unsigned long long)e.tracker->num_frees,
                (unsigned long long)e.tracker->num_hwm,
                (unsigned long long)(e.tracker->num_allocs - e.tracker->num_frees),
                format_size(e.tracker->total_alloc).c_str(),
                format_size(e.tracker->total_hwm).c_str(),
                format_size(e.tracker->total_alloc-e.tracker->total_free).c_str()
            );
            if (n > 0) {
                report.append(linebuf, static_cast<size_t>(n));
            }
        }
    }
    report += "Block Allocator Stats:\n";
    for (auto &e : registry) {
        if (e.block_allocator!=0)
        {
            int n = std::snprintf(
                linebuf, sizeof(linebuf),
                "  %30s : fix sz=%8llu outst=%6llu : super total=%8llu free=%14llu hwm=%14llu : var outst=%6llu mem=%8lld mwmhwm=%8lld\n",
                e.name.c_str(),
                    (unsigned long long)e.block_allocator->block_size,
                    (unsigned long long)e.block_allocator->allocated_blocks,
                    (unsigned long long)e.block_allocator->total_big_blocks,
                    (unsigned long long)e.block_allocator->free_big_blocks,
                    (unsigned long long)e.block_allocator->high_watermark_big_blocks,
                    (unsigned long long)e.block_allocator->total_variable_blocks,
                    (unsigned long long)e.block_allocator->total_variable_memory,
                    (unsigned long long)e.block_allocator->total_variable_memory_hwm
                );
            if (n > 0) {
                report.append(linebuf, static_cast<size_t>(n));
            }
        }
    }

    // Return a strdup'ed C string; caller must free().
    return strdup(report.c_str());
}

static inline void remove_from_free_list(block_allocator_t *ba, big_block_t *bb) {
    if (bb->prev_free) bb->prev_free->next_free = bb->next_free;
    else ba->free_list_head = bb->next_free;
    if (bb->next_free) bb->next_free->prev_free = bb->prev_free;
    bb->next_free = bb->prev_free = NULL;
}

static inline void add_to_free_list(block_allocator_t *ba, big_block_t *bb) {
    bb->next_free = ba->free_list_head;
    if (ba->free_list_head) ba->free_list_head->prev_free = bb;
    ba->free_list_head = bb;
    bb->prev_free = NULL;
}

static big_block_t *big_block_create(block_allocator_t *ba) {
    size_t hdr_sz = sizeof(big_block_t);
    size_t sub_hdr_sz = sizeof(block_header_t);
    size_t single_sz = sub_hdr_sz + ba->block_size;
    size_t total_sz = hdr_sz + BLOCKS_PER_BIG * single_sz;
    void *mem = malloc(total_sz);
    if (!mem) return NULL;
    big_block_t *bb = (big_block_t *)mem;
    bb->canary = BIG_CANARY;
    bb->block_size = ba->block_size;
    bb->free_mask = UINT64_MAX;
    bb->next_free = bb->prev_free = NULL;
    bb->next_all = bb->prev_all = NULL;
    bb->allocator = ba;

    uint8_t *p = (uint8_t*)mem + hdr_sz;
    for (int i = 0; i < BLOCKS_PER_BIG; i++) {
        block_header_t *bh = (block_header_t*)p;
        bh->parent = bb;
        bh->free_callback = NULL;
        bh->callback_context = NULL;
        bh->ref_count = 0;
        bh->canary = SUB_CANARY;
        bh->idx = i;
        bh->block_size=(unsigned int)ba->block_size;
        bh->block_allocator=ba;
        p += single_sz;
    }
    return bb;
}

block_allocator_t *block_allocator_create(const char *name, size_t block_size) {
    block_allocator_t *ba = (block_allocator_t *)malloc(sizeof(*ba));
    if (!ba) return NULL;
    memset(ba, 0, sizeof(block_allocator_t));
    ba->block_size = block_size;
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&ba->lock, &attr);
    ba->canary=BA_CANARY;
    block_allocator_register(ba, name);
    return ba;
}

void block_allocator_destroy(block_allocator_t *ba) {
    if (!ba) return;
    pthread_mutex_lock(&ba->lock);
    big_block_t *bb = ba->all_blocks;
    while (bb) {
        big_block_t *next = bb->next_all;
        free(bb);
        bb = next;
    }
    pthread_mutex_unlock(&ba->lock);
    pthread_mutex_destroy(&ba->lock);
    free(ba);
}

void *block_alloc(block_allocator_t *ba, size_t size) {
    if (size<=ba->block_size)
    {
        return block_alloc(ba);
    }

    block_header_t *bh=(block_header_t *)malloc(sizeof(block_header_t)+size);
    void *mem=(void *)(bh+1);
    bh->block_allocator=ba;
    bh->block_size=(unsigned int)size;
    bh->canary=SUB_CANARY;
    bh->parent=0;
    bh->free_callback=0;
    bh->callback_context=0;
    bh->ref_count=1;
    bh->idx=0;

    __sync_fetch_and_add(&ba->total_variable_memory, size);
    __sync_fetch_and_add(&ba->total_variable_blocks, 1);
    if (ba->total_variable_memory>ba->total_variable_memory_hwm)
        ba->total_variable_memory_hwm=ba->total_variable_memory;
    return mem;
}

void *block_alloc(block_allocator_t *ba) {
    pthread_mutex_lock(&ba->lock);
    big_block_t *bb = ba->free_list_head;
    if (!bb) {
        bb = big_block_create(ba);
        if (!bb) { pthread_mutex_unlock(&ba->lock); return NULL; }
        bb->next_all = ba->all_blocks;
        if (ba->all_blocks) ba->all_blocks->prev_all = bb;
        ba->all_blocks = bb;
        ba->total_big_blocks++;
        if (ba->total_big_blocks > ba->high_watermark_big_blocks)
            ba->high_watermark_big_blocks = ba->total_big_blocks;
        add_to_free_list(ba, bb);
    }
    uint64_t mask = bb->free_mask;
    int idx = __builtin_ctzll(mask);
    uint64_t bit = 1ULL << idx;
    bb->free_mask &= ~bit;
    if (bb->free_mask == 0) remove_from_free_list(ba, bb);

    size_t sub_hdr_sz = sizeof(block_header_t);
    size_t single_sz = sub_hdr_sz + ba->block_size;
    block_header_t *bh = (block_header_t*)((uint8_t*)bb + sizeof(big_block_t) + idx * single_sz);
    bh->ref_count = 1;
    bh->free_callback = NULL;
    bh->callback_context = NULL;

    ba->allocated_blocks++;
    pthread_mutex_unlock(&ba->lock);
    return (void*)((uint8_t*)bh + sub_hdr_sz);
}

void *block_reference(void *block) {
    if (!block) return NULL;
    block_header_t *bh = (block_header_t*)((uint8_t*)block - sizeof(block_header_t));
    assert(bh->canary == SUB_CANARY);
    assert(bh->ref_count>=1);
    __sync_fetch_and_add(&bh->ref_count, 1);
    return block;
}

int block_reference_count(void *block)
{
    block_header_t *bh = (block_header_t*)((uint8_t*)block - sizeof(block_header_t));
    return (int)__sync_fetch_and_add(&bh->ref_count, 0);
}

void block_free(void *block) {
    if (!block) return;
    block_header_t *bh = (block_header_t*)((uint8_t*)block - sizeof(block_header_t));
    big_block_t *bb = bh->parent;
    block_allocator_t *ba = bh->block_allocator;
    assert(bh->canary == SUB_CANARY);
    assert(ba->canary == BA_CANARY);

    int old_reference_count=__sync_fetch_and_sub(&bh->ref_count, 1);
    assert(old_reference_count>=1);
    if (old_reference_count>1) return;

    // last reference

    if (bh->free_callback) bh->free_callback(bh->callback_context, block);

    if (bb==0)
    {
        // variable sized block
        __sync_fetch_and_sub(&ba->total_variable_memory, bh->block_size);
        __sync_fetch_and_sub(&ba->total_variable_blocks, 1);
        bh->canary=SUB_CANARY_DEAD;
        free(bh);
        return;
    }

    // this is a fixed size block
    // careful when locking, but I had a long chat with chatGPT as to why this is safe
    pthread_mutex_lock(&ba->lock);
    assert(bb->canary == BIG_CANARY);
    if (ba->allocated_blocks > 0)
        ba->allocated_blocks--;

    uint64_t bit = 1ULL << bh->idx;

    big_block_t *to_free = NULL;
    uint64_t prev_mask = bb->free_mask;
    bb->free_mask |= bit;
    if (prev_mask == 0) {
        add_to_free_list(ba, bb);
    }
    if (bb->free_mask == UINT64_MAX) {
        remove_from_free_list(ba, bb);
        ba->free_big_blocks++;
        if (ba->free_big_blocks > ba->total_big_blocks/2) {
            // unlink from all_blocks
            if (bb->prev_all) bb->prev_all->next_all = bb->next_all;
            else ba->all_blocks = bb->next_all;
            if (bb->next_all) bb->next_all->prev_all = bb->prev_all;
            ba->total_big_blocks--;
            ba->free_big_blocks--;
            to_free = bb;
        }
    }
    pthread_mutex_unlock(&ba->lock);

    if (to_free) free(to_free);
}

void block_set_free_callback(void *block, void *context, void (*free_callback)(void *context, void *block)) {
    if (!block) return;
    block_header_t *bh = (block_header_t*)((uint8_t*)block - sizeof(block_header_t));
    assert(bh->canary == SUB_CANARY);
    assert(bh->free_callback==0);
    bh->free_callback = free_callback;
    bh->callback_context = context;
}

void block_check(void *block)
{
    block_header_t *bh = (block_header_t*)((uint8_t*)block - sizeof(block_header_t));
    assert(bh->canary == SUB_CANARY);
    assert(bh->ref_count>=1);
}
