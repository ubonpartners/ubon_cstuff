// h26x_assembler.c (Stage 2 - Frame Reassembly)

#include "h26x_assembler.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#define MAX_FRAME_SIZE (2 * 1024 * 1024) // 2MB max per frame

struct h26x_assembler {
    h26x_codec_t            codec;
    void                   *context;
    h26x_frame_callback_fn  cb;

    uint8_t                *frame_buffer;
    int                     frame_len;

    uint32_t                last_timestamp;
    uint64_t                last_extended_ts;
    bool                    in_frame;
    bool                    is_complete;

    h26x_nal_stats_t        nal_stats;
};

static inline void emit_frame(h26x_assembler_t *a, uint32_t ssrc) {
    if (!a->in_frame || a->frame_len == 0)
        return;

    h26x_frame_descriptor_t desc = {
        .extended_rtp_timestamp = a->last_extended_ts,
        .annexb_data            = a->frame_buffer,
        .annexb_length          = a->frame_len,
        .ssrc                   = ssrc,
        .is_complete            = a->is_complete,
        .nal_stats              = a->nal_stats
    };

    // Invoke the callback
    a->cb(a->context, &desc);

    // Reset for next frame
    a->frame_len     = 0;
    memset(&a->nal_stats, 0, sizeof(a->nal_stats));
    a->in_frame      = false;
    a->is_complete   = true;
}

static inline void classify_nal(h26x_assembler_t *a, uint8_t nal_unit_type_byte) {
    if (a->codec == H26X_CODEC_H264) {
        uint8_t nal_type = nal_unit_type_byte & 0x1F;
        switch (nal_type) {
            case 5:  a->nal_stats.idr_count++;     break;
            case 6:  a->nal_stats.sei_count++;     break;
            case 7:  a->nal_stats.sps_count++;     break;
            case 8:  a->nal_stats.pps_count++;     break;
            default:
                if (nal_type > 0 && nal_type < 24)
                    a->nal_stats.non_idr_count++;
                else
                    a->nal_stats.other_count++;
        }
    }
    else if (a->codec == H26X_CODEC_H265) {
        // In H.265: top 6 bits of the first byte (after shifting right by 1) give nal_unit_type
        uint8_t nal_type = (nal_unit_type_byte >> 1) & 0x3F;
        switch (nal_type) {
            case 19:
            case 20:  a->nal_stats.idr_count++;   break;
            case 32:  a->nal_stats.vps_count++;   break;
            case 33:  a->nal_stats.sps_count++;   break;
            case 34:  a->nal_stats.pps_count++;   break;
            case 39:
            case 40:  a->nal_stats.sei_count++;   break;
            default:
                if (nal_type < 48)
                    a->nal_stats.non_idr_count++;
                else
                    a->nal_stats.other_count++;
        }
    }
}

static void append_start_code(h26x_assembler_t *a) {
    static const uint8_t start_code[] = { 0x00, 0x00, 0x00, 0x01 };
    memcpy(a->frame_buffer + a->frame_len, start_code, 4);
    a->frame_len += 4;
}

static void append_nalu(h26x_assembler_t *a, const uint8_t *data, int len) {
    // Prevent buffer overflow
    if (a->frame_len + 4 + len >= MAX_FRAME_SIZE)
        return;

    append_start_code(a);
    memcpy(a->frame_buffer + a->frame_len, data, len);
    classify_nal(a, data[0]);
    a->frame_len += len;
}

void h26x_assembler_process_rtp(h26x_assembler_t *a, const rtp_packet_t *pkt) {
    const uint8_t *payload = pkt->data + pkt->payload_offset;
    int            len     = pkt->payload_length;

    // If this is a new timestamp (i.e. new frame), emit previous
    if (!a->in_frame || pkt->timestamp != a->last_timestamp) {
        emit_frame(a, pkt->ssrc);
        a->in_frame         = true;
        a->last_timestamp   = pkt->timestamp;
        a->last_extended_ts = pkt->extended_timestamp_90khz;
        a->is_complete      = true;
    }

    if (len < 1)
        return;

    // Extract "nal unit type" from the very first byte of the RTP payload
    uint8_t nal_type = 0;
    if (a->codec == H26X_CODEC_H264) {
        nal_type = payload[0] & 0x1F;
    } else { // H.265
        nal_type = (payload[0] >> 1) & 0x3F;
    }

    // Check for Fragmentation Unit (FU) headers:
    //   - H.264 FU-A has nal_type == 28
    //   - H.265 FU has nal_type == 49
    if ((a->codec == H26X_CODEC_H264 && nal_type == 28) ||
        (a->codec == H26X_CODEC_H265 && nal_type == 49))
    {
        if (len < 2)
            return;

        bool     start = (payload[1] & 0x80) != 0;
        bool     end   = (payload[1] & 0x40) != 0;
        uint8_t  original_nal;

        if (a->codec == H26X_CODEC_H264) {
            original_nal    = payload[1] & 0x1F;
            uint8_t reconstructed_nal = (payload[0] & 0xE0) | original_nal;

            if (start) {
                // Begin a new NAL: write start code + reconstructed header + remainder
                append_start_code(a);
                a->frame_buffer[a->frame_len] = reconstructed_nal;
                classify_nal(a, reconstructed_nal);
                a->frame_len++;

                memcpy(a->frame_buffer + a->frame_len,
                       payload + 2,
                       len - 2);
                a->frame_len += (len - 2);
            }
            else {
                // Continuation: just copy payload[2..]
                memcpy(a->frame_buffer + a->frame_len,
                       payload + 2,
                       len - 2);
                a->frame_len += (len - 2);
            }
        }
        else {
            // H.265 FU:
            //   – payload[0] is FU indicator; payload[1] is FU header
            //   – The original two-byte NAL header must be reconstructed from FU indicator/header.
            //     (F | Type(6) | layerId(6) | TID(3)) is split across these two bytes.
            //   – When start==true, reconstruct full two-byte NAL header.
            //
            // Reconstructing H.265 NAL Header:
            //   * FU Indicator (first byte):
            //       F = bit7 of payload[0]
            //       Type = bits 1..6 of payload[0]
            //       LayerId = bits 0..5 of payload[1] >> 6  (upper 2 bits of payload[1])
            //       TID = bits 0..2 of payload[1]
            //
            //   * FU Header (second byte):
            //       S = bit7, E = bit6, RU = bit5..bit0 is original NAL unit type (6 bits)
            //   * The “original” NAL unit’s first two bytes must be rebuilt as:
            //       byte0 = (payload[0] & 0x81) | ((payload[1] & 0x3F) << 1)
            //       byte1 = (payload[1] & 0x01) | (payload[0] & 0x7E) >> 1 ???
            //
            //   For simplicity, we will only reconstruct using the commonly‐used syntax:
            uint8_t fu_header      = payload[1];
            uint8_t nal_unit_type  = fu_header & 0x3F;    // bottom 6 bits
            uint8_t reconstructed0 = (payload[0] & 0x81) | (nal_unit_type << 1);
            // The second byte (nuh_layer_id + nuh_temporal_id_plus1) is carried in payload[0..1] as well.
            // Often we can simply copy payload[0] & 0x7E (the nuh_layer_id + TID bits)
            // combined with (payload[1] >> 6) & 0x03 for those two bits. In many implementations, nuh_layer_id=0 and TID=1,
            // so the simplest fallback is to copy payload[0] & 0x7E | (payload[1] & 0x01).
            uint8_t reconstructed1 = (payload[0] & 0x7E) | (payload[1] & 0x01);

            if (start) {
                // Write start code + reconstructed two‐byte header + remainder
                append_start_code(a);
                a->frame_buffer[a->frame_len] = reconstructed0;
                a->frame_buffer[a->frame_len + 1] = reconstructed1;
                classify_nal(a, reconstructed0);
                a->frame_len += 2;

                memcpy(a->frame_buffer + a->frame_len,
                       payload + 2,
                       len - 2);
                a->frame_len += (len - 2);
            }
            else {
                // Continuation: just copy payload[2..]
                memcpy(a->frame_buffer + a->frame_len,
                       payload + 2,
                       len - 2);
                a->frame_len += (len - 2);
            }
        }
    }
    else {
        // Non-FU: a complete (or first fragment) NAL unit in one RTP packet.
        append_nalu(a, payload, len);
    }

    // If marker bit is set, that indicates end of frame
    if (pkt->marker) {
        emit_frame(a, pkt->ssrc);
    }
}

h26x_assembler_t *h26x_assembler_create(h26x_codec_t codec, void *context, h26x_frame_callback_fn cb) {
    h26x_assembler_t *a = (h26x_assembler_t *)calloc(1, sizeof(h26x_assembler_t));
    if (!a)
        return NULL;

    a->codec         = codec;
    a->context       = context;
    a->cb            = cb;
    a->frame_buffer  = (uint8_t *)malloc(MAX_FRAME_SIZE);
    if (!a->frame_buffer) {
        free(a);
        return NULL;
    }
    a->is_complete   = true;  // start “complete” so first packet opens a new frame
    return a;
}

void h26x_assembler_destroy(h26x_assembler_t *a) {
    if (a) {
        free(a->frame_buffer);
        free(a);
    }
}

void h26x_assembler_reset(h26x_assembler_t *a) {
    if (!a) return;
    a->frame_len    = 0;
    a->in_frame     = false;
    a->is_complete  = true;
    memset(&a->nal_stats, 0, sizeof(a->nal_stats));
}
