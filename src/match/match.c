#include <stdint.h>
#include <cassert>
#include "string.h"
#include <math.h>
#include <stdio.h>
#include "match.h"

/* Pair of uint16 indices */
typedef struct {
    uint16_t a, b;
} Pair16;

#define MAX_ITEMS       65535

static int match_masks_optimized(
    const uint64_t *maskA,
    const uint64_t *maskB,
    int             nA,
    int             nB,
    Pair16          *out_pairs
) {
    if (nA <= 0 || nB <= 0 || nA > MAX_ITEMS || nB > MAX_ITEMS) {
        return 0;
    }

    uint16_t b_count[64];
    uint16_t b_indices[64][nB];
    uint16_t seen[nB];

    memset(b_count, 0, sizeof(uint16_t)*64);
    memset(b_indices, 0, sizeof(uint16_t)*64*nB);

    for (int j = 0; j < nB; ++j) {
        uint64_t mb = maskB[j];
        while (mb) {
            int b = __builtin_ctzll(mb);
            mb &= (mb - 1);
            uint16_t cnt = b_count[b];
            assert(cnt<=nB);
            b_indices[b][cnt] = (uint16_t)j;
            b_count[b] = cnt + 1;
        }
    }

    int out_idx = 0;
    for (int i = 0; i < nA; ++i) {
        memset(seen, 0, nB*sizeof(uint16_t));
        uint64_t ma = maskA[i];
        if (ma == 0ULL) continue;
        while (ma) {
            int b = __builtin_ctzll(ma);
            ma &= (ma - 1);
            uint16_t cnt = b_count[b];
            for (uint16_t k = 0; k < cnt; ++k) {
                uint16_t j = b_indices[b][k];
                if (!seen[j]) {
                    seen[j] = 1;
                    out_pairs[out_idx].a = (uint16_t)i;
                    out_pairs[out_idx].b = j;
                    ++out_idx;
                }
            }
        }
    }

    return out_idx;
}

typedef struct {
    uint16_t a, b;
    float   score;
} Cand;

// Comparator for qsort: descending by score
static int compare_cand_desc(const void *p1, const void *p2) {
    const Cand *c1=((const Cand*)p1);
    const Cand *c2=((const Cand*)p2);
    float s1 = c1->score-c1->a;
    float s2 = c2->score-c2->a;

    if (s1 < s2) return  1;
    if (s1 > s2) return -1;
    return 0;
}

int match_greedy(
    const void        **a,
    uint64_t          *maskA,
    int                num_a,
    const void        **b,
    uint64_t          *maskB,
    int                num_b,
    float            (*cost_fn)(const void *item_a, const void *item_b, void *context),
    void              *ctx,
    uint16_t           *out_a_idx,
    uint16_t           *out_b_idx,
    float              *out_score,
    bool               do_debug
)
{
    assert(num_a >= 0 && num_a <= MAX_ITEMS);
    assert(num_b >= 0 && num_b <= MAX_ITEMS);

    if (num_a <= 0 || num_b <= 0) {
        return 0;
    }

    // 1) Build sparse mask‐filtered list of candidate pairs
    Pair16 raw_cands[num_a * num_b];
    int num_raw = match_masks_optimized(
        maskA, maskB,
        num_a, num_b,
        raw_cands
    );
    assert(num_raw <= num_a * num_b);

    // 2) Compute each pair's score
    Cand *all = (Cand *)malloc(sizeof(Cand) * num_raw);
    if (!all) return 0;
    for (int i = 0; i < num_raw; ++i) {
        uint16_t ia = raw_cands[i].a;
        uint16_t ib = raw_cands[i].b;
        float sc = cost_fn(a[ia], b[ib], ctx);
        all[i].a     = ia;
        all[i].b     = ib;
        all[i].score = sc;
    }

    // 3) Sort all candidates-
    //   - by descending detection order first (=original sort order, usually descending confidence)
    //   - by descending IOU score second
    qsort(all, num_raw, sizeof(Cand), compare_cand_desc);

    // 4) Greedily pick top‐scoring non‐conflicting pairs
    uint8_t matchedA[num_a] = {0};
    uint8_t matchedB[num_b] = {0};
    int out_count = 0;

    for (int k = 0; k < num_raw; ++k) {
        uint16_t ia = all[k].a;
        uint16_t ib = all[k].b;
        float    sc = all[k].score;

        if (sc <= 0.0f)          continue;  // no benefit
        if (matchedA[ia] || matchedB[ib]) continue;

        matchedA[ia] = 1;
        matchedB[ib] = 1;
        out_a_idx[out_count] = ia;
        out_b_idx[out_count] = ib;
        if (out_score) out_score[out_count] = sc;
        ++out_count;
    }

    free(all);
    return out_count;
}
