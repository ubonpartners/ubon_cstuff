#ifndef __MATCH_PRIV_H
#define __MATCH_PRIV_H

#include <stdint.h>

/* Pair of uint16 indices */
typedef struct {
    uint16_t a, b;
} Pair16;

typedef struct {
    uint16_t a, b;
    float   score;
} Cand;

#define MAX_ITEMS       65535

int match_masks_optimized(
    const uint64_t *maskA,
    const uint64_t *maskB,
    int             nA,
    int             nB,
    Pair16          *out_pairs
);

#endif
