// h26x_assembler.h

#ifndef H26X_ASSEMBLER_H
#define H26X_ASSEMBLER_H

#include <stdint.h>
#include <stdbool.h>
#include "rtp_receiver.h"

// -----------------------------------------------------------------------------
// Supported codec IDs
// -----------------------------------------------------------------------------
typedef enum {
    H26X_CODEC_H264 = 0,
    H26X_CODEC_H265 = 1
} h26x_codec_t;

// -----------------------------------------------------------------------------
// NAL‐unit statistics (counts of various NAL types seen in one frame)
// -----------------------------------------------------------------------------
typedef struct {
    uint32_t idr_count;       // IDR slices (H.264) or key‐picture NALs (H.265)
    uint32_t sei_count;       // SEI messages
    uint32_t sps_count;       // SPS units
    uint32_t pps_count;       // PPS units
    uint32_t vps_count;       // (H.265 only) VPS units
    uint32_t non_idr_count;   // All other “normal” VCL NALs
    uint32_t other_count;     // Any NALs not counted above
    uint32_t depacketization_errors;  // count of malformed‐RTP or buffer‐overflow cases
} h26x_nal_stats_t;

// -----------------------------------------------------------------------------
// When a complete Annex‐B frame is assembled, we deliver it via this struct.
// -----------------------------------------------------------------------------
typedef struct {
    uint64_t            extended_rtp_timestamp;  // 90 kHz timestamp from the first RTP packet
    uint8_t            *annexb_data;             // Pointer to raw Annex‐B buffer (0x00000001 …)
    int                 annexb_length;           // Number of bytes in annexb_data
    uint32_t            ssrc;                    // SSRC of the RTP stream
    bool                is_complete;             // “true” if we believe no packet loss occurred
    h26x_nal_stats_t    nal_stats;               // Per‐frame NAL statistics
} h26x_frame_descriptor_t;

// -----------------------------------------------------------------------------
// User must supply a callback of this form; it is invoked whenever a full frame
// has been reassembled.  The callback should copy or otherwise “consume” the
// annexb_data buffer before returning (it’s volatile memory inside the assembler).
// -----------------------------------------------------------------------------
typedef void (*h26x_frame_callback_fn)(void *context,
                                       const h26x_frame_descriptor_t *desc);

// -----------------------------------------------------------------------------
// Opaque assembler context
// -----------------------------------------------------------------------------
typedef struct h26x_assembler h26x_assembler_t;

// -----------------------------------------------------------------------------
// Create a new assembler instance. ’codec’ selects H.264 vs. H.265.
// ’context’ is user‐defined and will be passed verbatim to every callback.
// ’cb’ is invoked once per reassembled frame (when RTP‐marker is seen).
// Returns NULL on failure.
// -----------------------------------------------------------------------------
h26x_assembler_t *h26x_assembler_create(h26x_codec_t            codec,
                                        void                   *context,
                                        h26x_frame_callback_fn  cb);

// -----------------------------------------------------------------------------
// Free all memory held by ’a’.  After this call, ’a’ is invalid.
// -----------------------------------------------------------------------------
void h26x_assembler_destroy(h26x_assembler_t *a);

// -----------------------------------------------------------------------------
// Reset internal state (discard any partially‐assembled frame).
// -----------------------------------------------------------------------------
void h26x_assembler_reset(h26x_assembler_t *a);

// -----------------------------------------------------------------------------
// Feed one RTP packet into the assembler.  If this packet completes a frame
// (either because timestamp changed or marker bit set), the user’s callback
// will be invoked.  All pointer fields inside ’pkt’ must remain valid while
// this call runs.
// -----------------------------------------------------------------------------
void h26x_assembler_process_rtp(h26x_assembler_t       *a,
                                const rtp_packet_t     *pkt);

// -----------------------------------------------------------------------------
// Second interface to feed raw annex-B video into the assembler
// The assembler extracts the raw nalus, bypasses the RTP assembly code and
// calls append_nalu. Unlike the RTP version it needs to detect end of frame
// by looking at the nalus to know to call emit_frame
// The assembler keeps an internal 90Khz timestamp and automatically increments
// it to set the output timestamp based on the framerate.
// The return value is the number of frames emitted as a result of this call.
// -----------------------------------------------------------------------------
int h26x_assembler_process_raw_video(h26x_assembler_t       *a,
                                     double framerate,
                                     const uint8_t *data,
                                     int data_len);

void h26x_print_frame_summary(const h26x_frame_descriptor_t *desc);

void h26x_assembler_fill_stats(h26x_assembler_t *a, h26x_nal_stats_t *stats);
void print_nal_stats(const h26x_nal_stats_t *stats);

#endif // H26X_ASSEMBLER_H
