#ifndef __MISC_H
#define __MISC_H

#include <stddef.h>
#include <stdint.h>

double time_now_sec();
void hash_2d(const uint8_t *mem, int w, int h, int stride, uint32_t *dest);
uint32_t hash_u32(const uint32_t *mem, int num);
const char *ubon_cstuff_get_version(void);

typedef struct allocation_tracker
{
    uint64_t num_allocs;
    uint64_t num_frees;
    uint64_t total_alloc;
    uint64_t total_free;
} allocation_tracker_t;

static inline void track_alloc(allocation_tracker_t *t, size_t sz)
{
    __atomic_add_fetch(&t->num_allocs, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&t->total_alloc, sz, __ATOMIC_RELAXED);
}

static inline void track_free(allocation_tracker_t *t, size_t sz)
{
    __atomic_add_fetch(&t->num_frees, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&t->total_free, sz, __ATOMIC_RELAXED);
}

void allocation_tracker_register(allocation_tracker_t *t, const char *name);
char *allocation_tracker_stats(void);
#endif
