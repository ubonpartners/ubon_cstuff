/*
 * Improved greedy “linear sum assignment” between two detection sets,
 * minimizing the number of candidate‐scans by bucketing candidates by A‐index.
 *
 * For each detection in A (in order 0..num_dets_a−1), we look only at its
 * precomputed list of candidate B‐indices whose overlap_mask intersects.
 * Among those B’s not yet matched, we call cost_fn once each, pick the best positive score,
 * and match. Each B can only be matched once.
 *
 * Steps:
 *   1) Extract overlap_mask arrays from dets_a and dets_b.
 *   2) Run match_masks_optimized to get all (i,j) pairs with overlap_mask intersection ≠ 0.
 *   3) Build an array-of-lists: for each i in [0..nA−1], collect all candidate B‐indices j into
 *      cand_list_by_A[i][0..cand_count_by_A[i]−1].
 *   4) Greedy match: for i = 0..nA−1, scan only cand_list_by_A[i], skip already‐matched B’s,
 *      call cost_fn(dets_a+i, dets_b+j, ctx), pick best positive; if found, record and mark matched.
 *
 * This avoids the O(num_candidates * num_dets_a) scan and reduces it to O(num_candidates + num_dets_a * avg_candidates_per_A).
 *
 * Assumptions:
 *   – num_dets_a, num_dets_b ≤ MAX_DETS (200).
 *   – detection_t has member `uint64_t overlap_mask`.
 *   – cost_fn returns >0.0f for a valid match, ≤0.0f to skip.
 *   – out_a_idx, out_b_idx arrays are length ≥ min(num_dets_a, num_dets_b).
 */

#include <stdint.h>
#include <string.h>   // memset
#include <assert.h>
#include <stdio.h>
#include "detections.h"

#define MAX_DETS 200

/* Pair of uint8 indices */
typedef struct {
    uint8_t a, b;
} Pair8;

/* ==================================================================== */
/* match_masks_optimized: build sparse candidate list (i,j) where masks overlap */
static void match_masks_optimized(
    const uint64_t *maskA,
    const uint64_t *maskB,
    int             nA,
    int             nB,
    Pair8          *out_pairs,
    int            *pOutCount
) {
    if (nA <= 0 || nB <= 0 || nA > MAX_DETS || nB > MAX_DETS) {
        *pOutCount = 0;
        return;
    }

    uint8_t b_count[64];
    uint8_t b_indices[64][MAX_DETS];
    uint8_t seen[MAX_DETS];

    memset(b_count, 0, sizeof(b_count));
    for (int j = 0; j < nB; ++j) {
        uint64_t mb = maskB[j];
        while (mb) {
            int b = __builtin_ctzll(mb);
            mb &= (mb - 1);
            uint8_t cnt = b_count[b];
            b_indices[b][cnt] = (uint8_t)j;
            b_count[b] = cnt + 1;
        }
    }

    int out_idx = 0;
    for (int i = 0; i < nA; ++i) {
        memset(seen, 0, (size_t)nB);
        uint64_t ma = maskA[i];
        if (ma == 0ULL) continue;
        while (ma) {
            int b = __builtin_ctzll(ma);
            ma &= (ma - 1);
            uint8_t cnt = b_count[b];
            for (uint8_t k = 0; k < cnt; ++k) {
                uint8_t j = b_indices[b][k];
                if (!seen[j]) {
                    seen[j] = 1;
                    out_pairs[out_idx].a = (uint8_t)i;
                    out_pairs[out_idx].b = j;
                    ++out_idx;
                }
            }
        }
    }

    *pOutCount = out_idx;
}

/* ==================================================================== */
/* Greedy matching using bucketed candidate lists */

void match_detections_greedy(
    const detection_t *dets_a,
    int                num_dets_a,
    const detection_t *dets_b,
    int                num_dets_b,
    float            (*cost_fn)(const detection_t *, const detection_t *, void *),
    void              *ctx,
    uint8_t           *out_a_idx,
    uint8_t           *out_b_idx,
    int               *pOutCount
) {
    assert(num_dets_a >= 0 && num_dets_a <= MAX_DETS);
    assert(num_dets_b >= 0 && num_dets_b <= MAX_DETS);

    if (num_dets_a <= 0 || num_dets_b <= 0) {
        *pOutCount = 0;
        return;
    }

    /* 1) Extract overlap_mask arrays */
    uint64_t maskA[MAX_DETS], maskB[MAX_DETS];
    for (int i = 0; i < num_dets_a; ++i) {
        maskA[i] = dets_a[i].overlap_mask;
    }
    for (int j = 0; j < num_dets_b; ++j) {
        maskB[j] = dets_b[j].overlap_mask;
    }

    /* 2) Build sparse candidate list (all pairs with mask intersection) */
    static Pair8 candidates[MAX_DETS * MAX_DETS];
    int num_candidates = 0;
    match_masks_optimized(
        maskA, maskB,
        num_dets_a, num_dets_b,
        candidates,
        &num_candidates
    );

    //printf("%d / %d\n",num_candidates,num_dets_a*num_dets_b);

    /* 3) Bucket candidates by A‐index: cand_list_by_A[i][0..cand_count_by_A[i]-1] = array of B‐indices */
    static uint8_t cand_list_by_A[MAX_DETS][MAX_DETS];
    uint16_t cand_count_by_A[MAX_DETS];  /* ≤ num_dets_b ≤ 200 */

    /* Initialize counts to zero */
    for (int i = 0; i < num_dets_a; ++i) {
        cand_count_by_A[i] = 0;
    }
    /* Fill buckets */
    for (int idx = 0; idx < num_candidates; ++idx) {
        uint8_t i = candidates[idx].a;
        uint8_t j = candidates[idx].b;
        uint16_t cnt = cand_count_by_A[i];
        cand_list_by_A[i][cnt] = j;
        cand_count_by_A[i] = cnt + 1;
    }

    /* 4) Greedy match */
    uint8_t matchedA[MAX_DETS] = {0};
    uint8_t matchedB[MAX_DETS] = {0};
    int out_count = 0;

    for (int i = 0; i < num_dets_a; ++i) {
        if (matchedA[i]) continue;  // should be false, by design

        float best_score = 0.0f;
        int best_j = -1;

        uint16_t cnt_i = cand_count_by_A[i];
        for (uint16_t k = 0; k < cnt_i; ++k) {
            uint8_t j = cand_list_by_A[i][k];
            if (matchedB[j]) continue;
            /* Only now call the expensive cost_fn */
            float c = cost_fn(&dets_a[i], &dets_b[j], ctx);
            if (c > best_score) {
                best_score = c;
                best_j = j;
            }
        }

        if (best_j >= 0 && best_score > 0.0f) {
            matchedA[i] = 1;
            matchedB[best_j] = 1;
            out_a_idx[out_count] = (uint8_t)i;
            out_b_idx[out_count] = (uint8_t)best_j;
            ++out_count;
        }
    }

    *pOutCount = out_count;
}

/* ==================================================================== */
/* Example usage:

#include <stdio.h>

// Simple cost function stub: positive if overlap_mask shares any bit
float my_cost_fn(const detection_t *x, const detection_t *y, void *ctx) {
    (void)ctx;
    return (x->overlap_mask & y->overlap_mask) ? 1.0f : 0.0f;
}

int main(void) {
    detection_t dets_a[3], dets_b[4];
    // Fill overlap_mask (example values)
    dets_a[0].overlap_mask = 0x0000000000000001ULL;  // bit 0
    dets_a[1].overlap_mask = 0x0000000000001000ULL;  // bit 12
    dets_a[2].overlap_mask = 0x8000000000000000ULL;  // bit 63

    dets_b[0].overlap_mask = 0x0000000000000001ULL;  // bit 0
    dets_b[1].overlap_mask = 0x0000000000002000ULL;  // bit 13
    dets_b[2].overlap_mask = 0x8000000000000000ULL;  // bit 63
    dets_b[3].overlap_mask = 0x0000000000000002ULL;  // bit 1 (no match)

    uint8_t outA[MAX_DETS], outB[MAX_DETS];
    int num_matches = 0;

    match_detections_greedy(
        dets_a, 3,
        dets_b, 4,
        my_cost_fn, NULL,
        outA, outB,
        &num_matches
    );

    printf("Matches (%d):\n", num_matches);
    for (int k = 0; k < num_matches; ++k) {
        printf("  A[%u] ↔ B[%u]\n", outA[k], outB[k]);
    }
    return 0;
}

*/
