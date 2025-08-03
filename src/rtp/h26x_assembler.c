// h26x_assembler.c

#include "h26x_assembler.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "log.h"

#define debugf if (0) log_error

#define MAX_FRAME_SIZE (2 * 1024 * 1024) // 2 MB max per frame

struct h26x_assembler {
    h26x_codec_t            codec;
    void                   *context;
    h26x_frame_callback_fn  cb;

    uint8_t                *frame_buffer;
    int                     frame_len;

    uint16_t                last_sequence_number;
    uint32_t                last_timestamp;
    uint64_t                last_extended_ts;
    bool                    in_frame;
    bool                    is_complete;
    bool                    frame_has_video_data;

    h26x_nal_stats_t        nal_stats;
    h26x_nal_stats_t        cum_nal_stats;

     // Fields for raw Annex-B processing
    uint8_t *raw_buffer;       // leftover bytes buffer
    int raw_buffer_len;        // number of bytes currently stored
    int raw_buffer_capacity;   // total allocated capacity
    double raw_ts;
};

/*
 * Emit the assembled frame (if any) via callback, then reset state for next frame.
 */
static void emit_frame(h26x_assembler_t *a, uint32_t ssrc) {
    debugf("Emit frame! %d",(int)a->frame_len);
    if (!a->in_frame || a->frame_len == 0) {
        return;
    }
    double t=a->last_extended_ts;
    h26x_frame_descriptor_t desc = {
        .extended_rtp_timestamp = a->last_extended_ts,
        .annexb_data            = a->frame_buffer,
        .annexb_length          = a->frame_len,
        .ssrc                   = ssrc,
        .is_complete            = a->is_complete,
        .nal_stats              = a->nal_stats
    };

    // Invoke user callback
    a->cb(a->context, &desc);

    // Reset for next frame
    a->frame_len     = 0;

    a->cum_nal_stats.idr_count+=a->nal_stats.idr_count;
    a->cum_nal_stats.sei_count+=a->nal_stats.sei_count;
    a->cum_nal_stats.sps_count+=a->nal_stats.sps_count;
    a->cum_nal_stats.pps_count+=a->nal_stats.pps_count;
    a->cum_nal_stats.vps_count+=a->nal_stats.vps_count;
    a->cum_nal_stats.non_idr_count+=a->nal_stats.non_idr_count;
    a->cum_nal_stats.other_count+=a->nal_stats.other_count;
    a->cum_nal_stats.depacketization_errors+=a->nal_stats.depacketization_errors;

    memset(&a->nal_stats, 0, sizeof(a->nal_stats));
    a->in_frame      = false;
    a->is_complete   = true;
    a->frame_has_video_data = false;
}

/*
 * Classify a single NAL unit by its first header‐byte, updating nal_stats.
 */
static inline void classify_nal(h26x_assembler_t *a, uint8_t nal_unit_header) {
    if (a->codec == H26X_CODEC_H264) {
        uint8_t nal_type = nal_unit_header & 0x1F;
        switch (nal_type) {
            case 5:  a->nal_stats.idr_count++; a->frame_has_video_data=true;   break;
            case 6:  a->nal_stats.sei_count++;     break;
            case 7:  a->nal_stats.sps_count++;     break;
            case 8:  a->nal_stats.pps_count++;     break;
            default:
                a->frame_has_video_data=true;
                if (nal_type > 0 && nal_type < 24)
                    a->nal_stats.non_idr_count++;
                else
                    a->nal_stats.other_count++;
        }
    }
    else { // H.265
        uint8_t nal_type = (nal_unit_header >> 1) & 0x3F;
        switch (nal_type) {
            case 19:
            case 20:  a->nal_stats.idr_count++; a->frame_has_video_data=true;   break;
            case 32:  a->nal_stats.vps_count++;   break;
            case 33:  a->nal_stats.sps_count++;   break;
            case 34:  a->nal_stats.pps_count++;   break;
            case 39:
            case 40:  a->nal_stats.sei_count++;   break;
            default:
                a->frame_has_video_data=true;
                if (nal_type < 48)
                    a->nal_stats.non_idr_count++;
                else
                    a->nal_stats.other_count++;
        }
    }
}

/*
 * Write a 4‐byte Annex‐B start code (0x00 0x00 0x00 0x01) into frame_buffer.
 */
static void append_start_code(h26x_assembler_t *a) {
    static const uint8_t start_code[4] = { 0x00, 0x00, 0x00, 0x01 };
    memcpy(a->frame_buffer + a->frame_len, start_code, 4);
    a->frame_len += 4;
}

/*
 * Append a single, complete NAL unit (in Annex‐B form) to frame_buffer:
 *   - write start code
 *   - classify it (increment stats)
 *   - copy raw NAL bytes
 */
static void append_nalu(h26x_assembler_t *a, const uint8_t *data, int len) {
    // Bound check: need 4 bytes for start code + len bytes for NAL
    if (a->frame_len + 4 + len >= MAX_FRAME_SIZE) {
        a->nal_stats.depacketization_errors++;
        return;
    }

    append_start_code(a);
    classify_nal(a, data[0]);
    memcpy(a->frame_buffer + a->frame_len, data, len);
    a->frame_len += len;
}

/*
 * Entry point: process one RTP packet’s payload, assemble into Annex‐B.
 */
void h26x_assembler_process_rtp(h26x_assembler_t *a, const rtp_packet_t *pkt) {
    const uint8_t *payload = pkt->data + pkt->payload_offset;
    int            len     = pkt->payload_length;

    debugf("assembler packet seq %d len %d",pkt->sequence_number,pkt->payload_length);

    // If this packet carries a new timestamp, close out the previous frame first
    if (!a->in_frame || pkt->timestamp != a->last_timestamp) {
        emit_frame(a, pkt->ssrc);
        a->in_frame         = true;
        a->last_timestamp   = pkt->timestamp;
        a->last_extended_ts = pkt->extended_timestamp;
        a->is_complete      = true;
        a->last_sequence_number = (uint16_t)((pkt->sequence_number - 1) & 0xFFFF);
    }

    if (len < 1) {
        // No payload at all → count as an error
        a->nal_stats.depacketization_errors++;
        return;
    }

    // Mark completeness: if sequence is not exactly last_seq+1, mark incomplete
    bool seq_ok = (pkt->sequence_number == (uint16_t)((a->last_sequence_number + 1) & 0xFFFF));
    a->is_complete &= seq_ok;
    a->last_sequence_number = pkt->sequence_number;

    // Determine raw NAL-unit type from the first payload byte
    uint8_t nal_type;
    if (a->codec == H26X_CODEC_H264) {
        nal_type = payload[0] & 0x1F;
    } else { // H.265
        nal_type = (payload[0] >> 1) & 0x3F;
    }

    // ───────────────────────────────────────────────────────────────────────────────
    // H.264: STAP-A (NAL type 24)
    // ───────────────────────────────────────────────────────────────────────────────
    if (a->codec == H26X_CODEC_H264 && nal_type == 24) {
        // Single‐Time Aggregation Packet: payload[0]=STAP-A header
        // Following: [16-bit size][NALU1][16-bit size][NALU2]...
        int offset = 1;
        while (offset + 2 <= len) {
            uint16_t nalu_size = (uint16_t)((payload[offset] << 8) | payload[offset + 1]);
            offset += 2;
            if (nalu_size == 0 || offset + nalu_size > len) {
                a->nal_stats.depacketization_errors++;
                break;
            }
            append_nalu(a, payload + offset, nalu_size);
            offset += nalu_size;
        }
    }
    // ───────────────────────────────────────────────────────────────────────────────
    // H.264: FU-A (NAL type 28)
    // ───────────────────────────────────────────────────────────────────────────────
    else if (a->codec == H26X_CODEC_H264 && nal_type == 28) {
        // Fragmentation Unit (FU-A):
        //   payload[0]=FU indicator, payload[1]=FU header, payload[2..]=fragment data
        if (len < 2) {
            a->nal_stats.depacketization_errors++;
            return;
        }
        bool start = (payload[1] & 0x80) != 0;
        bool end   = (payload[1] & 0x40) != 0;
        uint8_t original_nal_type = payload[1] & 0x1F;
        uint8_t reconstructed_header = (payload[0] & 0xE0) | original_nal_type;

        if (start) {
            // New fragmented NAL: write start code + reconstructed header + data
            int needed = 4 /*start code*/ + 1 /*header*/ + (len - 2);
            if (a->frame_len + needed >= MAX_FRAME_SIZE) {
                a->nal_stats.depacketization_errors++;
                return;
            }
            append_start_code(a);
            classify_nal(a, reconstructed_header);
            a->frame_buffer[a->frame_len++] = reconstructed_header;
            memcpy(a->frame_buffer + a->frame_len, payload + 2, (size_t)(len - 2));
            a->frame_len += (len - 2);
        }
        else {
            // Continuation: simply append payload[2..]
            int needed = (len - 2);
            if (a->frame_len + needed >= MAX_FRAME_SIZE) {
                a->nal_stats.depacketization_errors++;
                return;
            }
            memcpy(a->frame_buffer + a->frame_len, payload + 2, (size_t)(len - 2));
            a->frame_len += (len - 2);
        }
        // Note: even if end==true, we wait for marker bit to signal end‐of‐frame
    }
    // ───────────────────────────────────────────────────────────────────────────────
    // H.265: (exactly as your original code)—
    //   FU (type 49) and single‐NALU cases unchanged:
    // ───────────────────────────────────────────────────────────────────────────────
    else if (a->codec == H26X_CODEC_H265 && nal_type == 48) {
        // In H.265, the NAL header is 2 bytes. So payload[0..1] is the AP header:
        //   [ F | nal_unit_type=48 | layer_id(6) | TID(3) ]  <-- 2 bytes total
        // After those 2 header bytes, we have: [16-bit size][NALU1][16-bit size][NALU2]…

        int offset = 2; // skip the 2-byte AP header
        while (offset + 2 <= len) {
            // Read 16-bit big-endian NALU size
            uint16_t nalu_size = (uint16_t)((payload[offset] << 8) | payload[offset + 1]);
            offset += 2;

            // If size is zero or exceeds remaining payload, it’s malformed
            if (nalu_size == 0 || offset + nalu_size > len) {
                a->nal_stats.depacketization_errors++;
                return;
            }

            // Check buffer space: need 4 bytes for Annex‐B start code + nalu_size bytes
            if (a->frame_len + 4 + nalu_size >= MAX_FRAME_SIZE) {
                a->nal_stats.depacketization_errors++;
                return;
            }

            // Append start code + raw NALU
            append_start_code(a);
            classify_nal(a, payload[offset]); // first byte of that NALU
            memcpy(a->frame_buffer + a->frame_len,
                payload + offset,
                nalu_size);
            a->frame_len += nalu_size;
            offset += nalu_size;
        }

        // If we didn’t exactly consume all bytes, mark error
        if (offset != len) {
            a->nal_stats.depacketization_errors++;
            return;
        }
    }
    else if (a->codec == H26X_CODEC_H265 && nal_type == 49) {
        if (len < 3) {
            a->nal_stats.depacketization_errors++;
            return;
        }

        // --- 3-byte pdu header breakdown ---
        uint8_t payload_hdr0 = payload[0]; // F (1) | 49 (6) | layer_id_msb (1)
        uint8_t payload_hdr1 = payload[1]; // layer_id_lsb (5) | tid_plus1 (3)
        uint8_t fu_hdr      = payload[2]; // S | E | R | orig_nal_type (6 bits)

        bool start = (fu_hdr & 0x80) != 0;
        bool end   = (fu_hdr & 0x40) != 0;  // not used here except for debugging
        uint8_t orig_nal_type = (fu_hdr & 0x3F);
        if (start) {
            // Reconstruct the first two bytes of the original NuH:
            //  • Rebuilt NuH[0] = (F<<7) | (orig_nal_type<<1) | (layer_id_msb<<0)
            uint8_t reconstructed0 =
                  (payload_hdr0 & 0x80)        // keep the F bit
                | (orig_nal_type << 1)          // put orig_nal_type into bits [6:1]
                | (payload_hdr0 & 0x01);        // keep layer_id_MSB (bit 0)

            // Rebuilt NuH[1] = (layer_id_lsb << 3) | tid_plus1
            // But note: payload_hdr1 already is exactly that bit-pattern:
            //    payload_hdr1 = (layer_id_lsb<<3) | (tid_plus1),
            // so we can just copy it directly.
            uint8_t reconstructed1 = payload_hdr1;

            // Check buffer space: 4 (start code) + 2 (reconstructed NuH) + (len−3) (RBSP)
            int needed = 4 + 2 + (len - 3);
            if (a->frame_len + needed >= MAX_FRAME_SIZE) {
                a->nal_stats.depacketization_errors++;
                return;
            }

            append_start_code(a);
            a->frame_buffer[a->frame_len++] = reconstructed0;
            a->frame_buffer[a->frame_len++] = reconstructed1;
            classify_nal(a, reconstructed0);

            // Copy the RBSP payload (everything after the 3-byte FU header)
            memcpy(a->frame_buffer + a->frame_len,
                   payload + 3,
                   (size_t)(len - 3));
            a->frame_len += (len - 3);
        }
        else {
            // Middle or end fragment: just copy RBSP (payload[3..])
            int rbsp_len = len - 3;
            if (rbsp_len < 0 || a->frame_len + rbsp_len >= MAX_FRAME_SIZE) {
                a->nal_stats.depacketization_errors++;
                return;
            }
            memcpy(a->frame_buffer + a->frame_len,
                   payload + 3,
                   (size_t)rbsp_len);
            a->frame_len += rbsp_len;
        }
        // (We still wait for the marker bit to emit Frame; do NOT emit here.)
    }
    else if (a->codec == H26X_CODEC_H265) {
        // Non-FU H.265 single NAL:
        append_nalu(a, payload, len);
    }
    // ───────────────────────────────────────────────────────────────────────────────
    // Default (either H.264 single­NALU or H.265 single­NALU not caught above)
    // ───────────────────────────────────────────────────────────────────────────────
    else {
        append_nalu(a, payload, len);
    }

    // If the RTP marker bit is set, emit assembled frame now
    if (pkt->marker && a->frame_has_video_data) {
        emit_frame(a, pkt->ssrc);
    }
}

/*
 * Create a new assembler instance.
 */
h26x_assembler_t *h26x_assembler_create(h26x_codec_t codec, void *context, h26x_frame_callback_fn cb) {
    h26x_assembler_t *a = (h26x_assembler_t *)calloc(1, sizeof(h26x_assembler_t));
    if (!a) {
        return NULL;
    }

    a->codec         = codec;
    a->context       = context;
    a->cb            = cb;
    a->frame_buffer  = (uint8_t *)malloc(MAX_FRAME_SIZE);
    if (!a->frame_buffer) {
        free(a);
        return NULL;
    }
    a->frame_len     = 0;
    a->in_frame      = false;
    a->is_complete   = true;  // so the very first packet starts a new frame
    memset(&a->nal_stats, 0, sizeof(a->nal_stats));

    return a;
}

// Bit-reader for parsing Exp-Golomb
typedef struct {
    const uint8_t *buf;
    int            size_bytes;
    int            bit_pos;
} bs_t;

static void bs_init(bs_t *bs, const uint8_t *buf, int sz) {
    bs->buf = buf;
    bs->size_bytes = sz;
    bs->bit_pos = 0;
}

static int bs_read_bit(bs_t *bs) {
    if (bs->bit_pos >= bs->size_bytes * 8) return 0;
    int b = bs->buf[bs->bit_pos >> 3];
    int shift = 7 - (bs->bit_pos & 7);
    bs->bit_pos++;
    return (b >> shift) & 1;
}

// reads an unsigned Exp-Golomb code (ue(v))
static int bs_read_ue(bs_t *bs) {
    int zeros = 0;
    while (bs_read_bit(bs) == 0 && bs->bit_pos < bs->size_bytes * 8) {
        zeros++;
    }
    int value = 1;
    for (int i = 0; i < zeros; i++) {
        value = (value << 1) | bs_read_bit(bs);
    }
    return value - 1;
}

static int remove_emulation_bytes(const uint8_t *src, int src_len, uint8_t *dst) {
    int dst_len = 0;
    int zeros = 0;
    for (int i = 0; i < src_len; i++) {
        uint8_t b = src[i];
        if (zeros >= 2 && b == 0x03) {
            // skip
            zeros = 0;
            continue;
        }
        dst[dst_len++] = b;
        if (b == 0) zeros++; else zeros = 0;
    }
    return dst_len;
}

// H.264: first_mb_in_slice == 0 → first slice of picture
static bool h264_is_first_mb(const uint8_t *nalu, int len) {
    if (len <= 1) return false;
    const uint8_t *payload = nalu + 1;
    int payload_len = len - 1;

    uint8_t rbsp[16];
    int to_parse=payload_len<16 ? payload_len : 16;
    int rbsp_len = remove_emulation_bytes(payload, to_parse, rbsp);

    bs_t bs; bs_init(&bs, rbsp, rbsp_len);
    int first_mb = bs_read_ue(&bs);
    return (first_mb == 0);
}

// H.265: first_slice_segment_in_pic_flag == 1 → first slice of picture
static bool h265_is_first_slice(const uint8_t *nalu, int len) {
    if (len <= 2) return false;
    const uint8_t *payload = nalu + 2;
    int payload_len = len - 2;

    uint8_t rbsp[16];
    int to_parse=payload_len<16 ? payload_len : 16;
    int rbsp_len = remove_emulation_bytes(payload, to_parse, rbsp);

    bs_t bs; bs_init(&bs, rbsp, rbsp_len);
    int first_flag = bs_read_bit(&bs);
    return (first_flag == 1);
}

#define MAX_FRAME_SIZE (2 * 1024 * 1024)
#define INITIAL_RAW_CAPACITY (512 * 1024)

int h26x_assembler_process_raw_video(h26x_assembler_t *a,
                                     double frame_rate,
                                     const uint8_t *data,
                                     int data_len) {
    if (!a || data_len <= 0) return 0;

    // lazy buffer alloc/grow
    if (!a->raw_buffer) {
        a->raw_buffer = (uint8_t *)malloc(INITIAL_RAW_CAPACITY);
        if (!a->raw_buffer) return 0;
        a->raw_buffer_capacity = INITIAL_RAW_CAPACITY;
        a->raw_buffer_len = 0;
    }
    int needed = a->raw_buffer_len + data_len;
    if (needed > a->raw_buffer_capacity) {
        int new_cap = a->raw_buffer_capacity;
        while (new_cap < needed) new_cap <<= 1;
        uint8_t *nb = (uint8_t *)realloc(a->raw_buffer, new_cap);
        if (!nb) { a->raw_buffer_len = 0; return 0; }
        a->raw_buffer = nb;
        a->raw_buffer_capacity = new_cap;
    }
    memcpy(a->raw_buffer + a->raw_buffer_len, data, data_len);

    uint8_t *buf = a->raw_buffer;
    int      buf_len = a->raw_buffer_len + data_len;
    int      frames  = 0;

    // find start codes (optimized)
    int start_pos[1024], scn = 0;
    uint8_t *base = buf;
    uint8_t *p    = buf;
    uint8_t *end  = buf + buf_len;

    // we need at least 3 bytes for a 3-byte start code, and 4 for a 4-byte one
    while (p + 3 < end && scn < 1024) {
        // fast-forward to the next zero
        p = (uint8_t*)memchr(p, 0, end - p);
        if (!p || p + 3 >= end) break;

        // count how many zeros in a row (up to 3)
        int zeros = 1;
        if (p[1] == 0) {
            zeros++;
            if (p + 2 < end && p[2] == 0) zeros++;
        }

        // check for start code after those zeros
        if (zeros >= 2 && p + 3 < end && p[zeros] == 1) {
            // p points at the first zero of 00{0,1}01
            start_pos[scn++] = (int)(p - base);
            p += zeros + 1;   // skip past the 00…01
        } else {
            p += zeros;       // not a start code, skip the zeros we counted
        }
    }

    if (scn < 1) {
        a->raw_buffer_len = buf_len;
        return 0;
    }

    for (int i=0; i<scn-1; i++) {
        int pos  = start_pos[i];
        int next = start_pos[i+1];
        int sc_len = (buf[pos+2]==1 ? 3 : 4);
        uint8_t *nalu = buf + pos + sc_len;
        int      n_len = next - (pos + sc_len);
        if (n_len <= 0) continue;

        // nal_type
        int nal_type = (a->codec==H26X_CODEC_H264)
                     ? (nalu[0] & 0x1F)
                     : ((nalu[0] >> 1) & 0x3F);

        // check if this is a VCL NALU and the *first* slice of a picture
        bool is_vcl = (a->codec==H26X_CODEC_H264)
                    ? (nal_type >= 1 && nal_type <= 5)
                    : (nal_type >= 0 && nal_type <= 31);
        bool new_frame = false;
        if (is_vcl) {
            if (a->codec==H26X_CODEC_H264) {
                new_frame = h264_is_first_mb(nalu, n_len);
            } else {
                new_frame = h265_is_first_slice(nalu, n_len);
            }
        }

        // if we detect a new frame while already in one, emit the previous
        if (new_frame && a->in_frame) {
            emit_frame(a, 0);
            frames++;
            a->raw_ts     += 1.0 / frame_rate;
            a->in_frame    = false;
        }

        // start (or restart) frame state
        if (new_frame /*|| !a->in_frame*/) {
            a->in_frame       = true;
            a->is_complete    = true;
            a->last_extended_ts = (uint64_t)(90000.0 * a->raw_ts);
        }

        // accumulate this NALU
        append_nalu(a, nalu, n_len);
    }

    // keep leftover bytes
    int last = start_pos[scn-1];
    int leftover = buf_len - last;
    memmove(a->raw_buffer, buf + last, leftover);
    a->raw_buffer_len = leftover;
    return frames;
}

/*
 * Reset assembler to “no frame in progress.”
 */
void h26x_assembler_reset(h26x_assembler_t *a) {
    if (!a) return;
    a->frame_len    = 0;
    a->in_frame     = false;
    a->is_complete  = true;
    memset(&a->nal_stats, 0, sizeof(a->nal_stats));
}

/*
 * Destroy/free the assembler.
 */
void h26x_assembler_destroy(h26x_assembler_t *a) {
    if (a) {
        free(a->frame_buffer);
        if (a->raw_buffer) free(a->raw_buffer);
        free(a);
    }
}

static void print_nal_stats_struct(const h26x_nal_stats_t *stats)
{
    printf("Depacketization Errors: %u\n", stats->depacketization_errors);
    printf("  IDR / Keyframe NALs : %u\n", stats->idr_count);
    printf("  Non-IDR VCL NALs    : %u\n", stats->non_idr_count);
    printf("  SEI Units           : %u\n", stats->sei_count);
    printf("  SPS Units           : %u\n", stats->sps_count);
    printf("  PPS Units           : %u\n", stats->pps_count);
    printf("  VPS Units (H.265)   : %u\n", stats->vps_count);
    printf("  Other NAL Types     : %u\n", stats->other_count);
}

void h26x_assembler_fill_stats(h26x_assembler_t *a, h26x_nal_stats_t *stats)
{
    memcpy(stats, &a->cum_nal_stats, sizeof(h26x_nal_stats_t));
}

void print_nal_stats(const h26x_nal_stats_t *stats)
{
    printf("h26x assembler stats:\n");
    print_nal_stats_struct(stats);
}

/*
 * Print a summary of frame descriptor (for debugging).
 */
void h26x_print_frame_summary(const h26x_frame_descriptor_t *desc) {
    if (!desc) return;

    printf("=== H26X Frame Summary ===\n");
    printf("RTP Timestamp   : %llu (90 kHz)\n", (unsigned long long)desc->extended_rtp_timestamp);
    printf("AnnexB Size     : %d bytes\n", desc->annexb_length);
    printf("SSRC            : 0x%08X\n", desc->ssrc);
    printf("Complete Frame? : %s\n", desc->is_complete ? "Yes" : "No");
    print_nal_stats_struct(&desc->nal_stats);
    printf("=========================\n");
}
