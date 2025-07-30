#ifndef __MEMORY_STUFF_H
#define __MEMORY_STUFF_H

#include <stddef.h>
#include <yaml-cpp/yaml.h>

typedef struct allocation_table allocation_table_t;

typedef struct allocation_tracker
{
    uint64_t num_allocs;
    uint64_t num_frees;
    uint64_t num_hwm;
    uint64_t total_alloc;
    uint64_t total_free;
    uint64_t total_hwm;
    allocation_table_t *allocation_table;
} allocation_tracker_t;

allocation_table_t *allocation_table_create();
void allocation_table_insert(allocation_table_t *table, void *ptr, size_t size);
size_t allocation_table_remove(allocation_table_t *table, void *ptr);
void allocation_table_check_ptr(allocation_table_t *table, void *ptr);

static inline void track_alloc(allocation_tracker_t *t, size_t sz)
{
    __atomic_add_fetch(&t->num_allocs, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&t->total_alloc, sz, __ATOMIC_RELAXED);
    int64_t num_outstanding=t->num_allocs-t->num_frees;
    int64_t total_outstanding=t->total_alloc-t->total_free;
    if (num_outstanding>t->num_hwm) t->num_hwm=num_outstanding;
    if (total_outstanding>t->total_hwm) t->total_hwm=total_outstanding;
}

static inline void track_alloc_table(allocation_tracker_t *t, size_t sz, void *p)
{
    allocation_table_insert(t->allocation_table, p, sz);
    track_alloc(t, sz);
}

static inline void track_check(allocation_tracker_t *t, void *p)
{
    if (t->allocation_table) allocation_table_check_ptr(t->allocation_table, p);
}

static inline void track_free(allocation_tracker_t *t, size_t sz)
{
    __atomic_add_fetch(&t->num_frees, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&t->total_free, sz, __ATOMIC_RELAXED);
}

static inline void track_free_table(allocation_tracker_t *t, void *p)
{
    size_t sz=allocation_table_remove(t->allocation_table, p);
    track_free(t, sz);
}

void allocation_tracker_register(allocation_tracker_t *t, const char *name, bool use_table=false);
char *allocation_tracker_stats(void);
YAML::Node allocation_tracker_stats_node();
void allocation_tracker_reset();
double allocation_tracker_get_mem_HWM(const char *tracker_name);
double allocation_tracker_get_mem_outstanding(const char *tracker_name);

typedef struct block_allocator block_allocator_t;

block_allocator_t *block_allocator_create(const char *name, size_t block_size);
void block_allocator_destroy(block_allocator_t *b);
void block_allocator_destroy(block_allocator_t *b);
uint32_t block_allocator_allocated_blocks(block_allocator_t *ba);
size_t block_allocator_block_size(block_allocator_t *ba);
void *block_alloc(block_allocator_t *b);
void *block_alloc(block_allocator_t *b, size_t sz);
void block_free(void *block);
void *block_reference(void *block);
void block_set_free_callback(void *block, void *context, void (*free_callback)(void *context, void *block));
void block_check(void *block);
int block_reference_count(void *block);

#endif
