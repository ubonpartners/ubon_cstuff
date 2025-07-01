#ifndef __MATCH_H
#define __MATCH_H

static inline uint64_t box_to_8x8_mask(float x0, float y0, float x1, float y1)
{
    /* 1) Reject degenerate or out‐of‐bounds boxes */
    if (x1 <= x0 || y1 <= y0 || x0 >= 1.0f || y0 >= 1.0f || x1 <= 0.0f || y1 <= 0.0f) {
        return 0ULL;
    }

    /* 2) Clamp the input to [0,1] to avoid any FP weirdness beyond [0,1] */
    if (x0 < 0.0f) x0 = 0.0f;
    if (y0 < 0.0f) y0 = 0.0f;
    if (x1 > 1.0f) x1 = 1.0f;
    if (y1 > 1.0f) y1 = 1.0f;

    /* 3) Convert normalized coords to [0..8) range */
    float fx0 = x0 * 8.0f;
    float fy0 = y0 * 8.0f;
    float fx1 = x1 * 8.0f;
    float fy1 = y1 * 8.0f;

    /* 4) Compute integer cell indices:
     *      c0 = floor(fx0)        (first column touched)
     *      c1 = ceil(fx1) - 1    (last column touched)
     *      r0 = floor(fy0)        (first row touched)
     *      r1 = ceil(fy1) - 1    (last row touched)
     *
     *    Because we treat an exact multiple of 1/8 as not entering the next cell,
     *    using ceil(fx1) - 1 handles that correctly: e.g. fx1=2.0 => ceil(2.0)-1 = 1.
     */
    int c0 = (int)floorf(fx0);
    int r0 = (int)floorf(fy0);
    int c1 = (int)ceilf(fx1) - 1;
    int r1 = (int)ceilf(fy1) - 1;

    /* 5) Clamp to valid cell indices [0..7] */
    if (c0 < 0) c0 = 0;   else if (c0 > 7) c0 = 7;
    if (r0 < 0) r0 = 0;   else if (r0 > 7) r0 = 7;
    if (c1 < 0) c1 = 0;   else if (c1 > 7) c1 = 7;
    if (r1 < 0) r1 = 0;   else if (r1 > 7) r1 = 7;

    /* 6) It may happen that after clamping, c1 < c0 or r1 < r0 (e.g. very thin box hugging a boundary),
     *    so double‐check and abort if there’s truly no overlap.
     */
    if (c1 < c0 || r1 < r0) {
        return 0ULL;
    }

    /* 7) Build a per‐row byte mask with bits [c0..c1] set.
     *      width = (c1 - c0 + 1)
     *    row_byte = ((1u << width) - 1) << c0
     */
    int width = c1 - c0 + 1;
    uint8_t row_byte = (uint8_t)(((1u << width) - 1u) << c0);

    /* 8) Replicate row_byte into every byte of a 64‐bit word:
     *    e.g. if row_byte = 0b00011100, then
     *         rep = 0x?[row_byte][row_byte][row_byte]…[row_byte]
     */
    const uint64_t BROADCAST = 0x0101010101010101ULL;
    uint64_t rep = (uint64_t)row_byte * BROADCAST;

    /* 9) Build a vertical mask that selects bytes r0..r1:
     *    We want (r1 - r0 + 1) consecutive bytes of 0xFF, starting at byte‐index r0.
     *    If we shift a 64‐bit all‐ones right by (64 - ((r1-r0+1)*8)), we get that many low bits set.
     *    Then shift left by (r0 * 8) to push those bytes into rows r0..r1.
     */
    int row_count = (r1 - r0 + 1);
    uint64_t vert_mask;
    if (row_count == 8) {
        /* special‐case: all rows */
        vert_mask = ~0ULL;
    } else {
        uint64_t low_bytes = (~0ULL) >> (64 - (row_count * 8));
        vert_mask = low_bytes << (r0 * 8);
    }

    /* 10) The final mask is the bitwise‐AND of rep and vert_mask */
    uint64_t mask = rep & vert_mask;
    return mask;
}

int match_greedy(
    const void        **a,
    uint64_t          *overlap_mask_a,
    int                num_a,
    const void        **b,
    uint64_t          *overlap_mask_b,
    int                num_b,
    float            (*cost_fn)(const void *item_a, const void *item_b, void *context),
    void              *ctx,
    uint16_t           *out_a_idx,
    uint16_t           *out_b_idx,
    float              *out_score=0,
    bool               do_debug=false
);

#endif
