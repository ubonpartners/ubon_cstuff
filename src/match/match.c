#include <stdint.h>
#include <cassert>
#include "string.h"
#include <math.h>
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
    uint16_t           *out_b_idx
)
{
    assert(num_a >= 0 && num_a <= MAX_ITEMS);
    assert(num_b >= 0 && num_b <= MAX_ITEMS);

    if (num_a <= 0 || num_b <= 0) {
        return 0;
    }

    /* 2) Build sparse candidate list (all pairs with mask intersection) */
    Pair16 candidates[num_a * num_b];
    int num_candidates = match_masks_optimized(
        maskA, maskB,
        num_a, num_b,
        candidates
    );
    assert(num_candidates<=num_a*num_b);

    //printf("%d / %d\n",num_candidates,num_a*num_b);

    /* 3) Bucket candidates by A‐index: cand_list_by_A[i][0..cand_count_by_A[i]-1] = array of B‐indices */
    uint16_t cand_list_by_A[num_a][num_b];
    uint16_t cand_count_by_A[num_a];  /* ≤ num_b ≤ 200 */

    /* Initialize counts to zero */
    memset(cand_count_by_A, 0, sizeof(uint16_t)*num_a);

    /* Fill buckets */
    for (int idx = 0; idx < num_candidates; ++idx) {
        uint16_t i = candidates[idx].a;
        uint16_t j = candidates[idx].b;
        uint16_t cnt = cand_count_by_A[i];
        cand_list_by_A[i][cnt] = j;
        assert(i<=num_a && cnt<=num_b);
        cand_count_by_A[i] = cnt + 1;
    }

    /* 4) Greedy match */
    uint16_t matchedA[num_a];
    uint16_t matchedB[num_b];
    int out_count = 0;
    memset(matchedA, 0, num_a*sizeof(int16_t));
    memset(matchedB, 0, num_b*sizeof(int16_t));

    for (int i = 0; i < num_a; ++i) {
        if (matchedA[i]) continue;  // should be false, by design

        float best_score = 0.0f;
        int best_j = -1;

        uint16_t cnt_i = cand_count_by_A[i];
        for (uint16_t k = 0; k < cnt_i; ++k) {
            uint16_t j = cand_list_by_A[i][k];
            assert(j<=num_b);
            if (matchedB[j]) continue;
            /* Only now call the expensive cost_fn */
            float c = cost_fn(a[i], b[j], ctx);
            if (c > best_score) {
                best_score = c;
                best_j = j;
            }
        }

        if (best_j >= 0 && best_score > 0.0f) {
            matchedA[i] = 1;
            matchedB[best_j] = 1;
            out_a_idx[out_count] = (uint16_t)i;
            out_b_idx[out_count] = (uint16_t)best_j;
            ++out_count;
        }
    }

    return out_count;
}
