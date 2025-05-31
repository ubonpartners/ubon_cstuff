// h26x_file_parser.c

#include "h26x_file_parser.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#define MAX_FRAME_SIZE   (2 * 1024 * 1024)  // 2 MB for one assembled frame
#define START_CODE_4LEN  4                  // 0x00 00 00 01
#define START_CODE_3LEN  3                  // 0x00 00 01

struct h26x_file_parser {
    h26x_codec_t            codec;
    void                   *context;
    h26x_frame_callback_fn  cb;

    uint8_t                *frame_buffer;
    int                     frame_len;

    uint64_t                next_timestamp_90k;   // 90 kHz ticks for next frame
    uint64_t                timestamp_increment;  // (90000 / framerate), rounded

    bool                    in_frame;             // true if we’ve started buffering an AU
    h26x_nal_stats_t        nal_stats;            // counts per‐frame
};

/**
 * Return 4 if data[pos..pos+3] == 0x00 00 00 01,
 * else 3 if data[pos..pos+2] == 0x00 00 01,
 * else 0.
 * Must check bounds externally before calling.
 */
static inline int get_start_code_length(const uint8_t *data, size_t pos, size_t length) {
    // Check for 4-byte first:
    if (pos + START_CODE_4LEN <= length &&
        data[pos]   == 0x00 &&
        data[pos+1] == 0x00 &&
        data[pos+2] == 0x00 &&
        data[pos+3] == 0x01)
    {
        return 4;
    }
    // Check for 3-byte:
    if (pos + START_CODE_3LEN <= length &&
        data[pos]   == 0x00 &&
        data[pos+1] == 0x00 &&
        data[pos+2] == 0x01)
    {
        return 3;
    }
    return 0;
}

/**
 * Classify a single NAL’s header byte into our nal_stats counters.
 * We re‐use the same logic as in h26x_assembler:
 */
static inline void classify_nal_header(h26x_file_parser_t *p, uint8_t nal_header_byte) {
    if (p->codec == H26X_CODEC_H264) {
        uint8_t nal_type = nal_header_byte & 0x1F;
        switch (nal_type) {
            case 5:  p->nal_stats.idr_count++;     break;
            case 6:  p->nal_stats.sei_count++;     break;
            case 7:  p->nal_stats.sps_count++;     break;
            case 8:  p->nal_stats.pps_count++;     break;
            default:
                if (nal_type > 0 && nal_type < 24)
                    p->nal_stats.non_idr_count++;
                else
                    p->nal_stats.other_count++;
        }
    }
    else { // H.265
        uint8_t nal_type = (nal_header_byte >> 1) & 0x3F;
        switch (nal_type) {
            case 19:
            case 20:  p->nal_stats.idr_count++;   break;
            case 32:  p->nal_stats.vps_count++;   break;
            case 33:  p->nal_stats.sps_count++;   break;
            case 34:  p->nal_stats.pps_count++;   break;
            case 39:
            case 40:  p->nal_stats.sei_count++;   break;
            default:
                if (nal_type < 48)
                    p->nal_stats.non_idr_count++;
                else
                    p->nal_stats.other_count++;
        }
    }
}

/**
 * Emit the currently‐buffered frame (access unit) via the user callback.
 * SSRC is always 0 (since this is a file parser).  is_complete = true always.
 */
static inline void emit_parsed_frame(h26x_file_parser_t *p) {
    if (!p->in_frame || p->frame_len == 0) {
        return; // nothing buffered
    }

    h26x_frame_descriptor_t desc = {
        .extended_rtp_timestamp = p->next_timestamp_90k,
        .annexb_data            = p->frame_buffer,
        .annexb_length          = p->frame_len,
        .ssrc                   = 0u,
        .is_complete            = true,
        .nal_stats              = p->nal_stats
    };

    // Invoke user callback
    p->cb(p->context, &desc);

    // Prepare for next frame
    p->next_timestamp_90k += p->timestamp_increment;
    p->frame_len = 0;
    memset(&p->nal_stats, 0, sizeof(p->nal_stats));
    p->in_frame = false;
}

/**
 * Append one raw NAL (including its 3- or 4-byte start code) into frame_buffer,
 * then update nal_stats.  If we exceed MAX_FRAME_SIZE, we drop.
 */
static inline void append_nal_to_frame(h26x_file_parser_t *p,
                                       const uint8_t      *nal_start_with_code,
                                       int                  nal_size,
                                       uint8_t             nal_header_byte)
{
    if (p->frame_len + nal_size > MAX_FRAME_SIZE) {
        // Drop if too large
        return;
    }
    memcpy(p->frame_buffer + p->frame_len, nal_start_with_code, nal_size);
    classify_nal_header(p, nal_header_byte);
    p->frame_len += nal_size;
}

/**
 * Public API: parse a full Annex-B buffer (H.264/H.265) as a series of access units.
 * Now detects both 4-byte (0x00000001) and 3-byte (0x000001) start codes.
 */
void h26x_file_parser_parse_buffer(h26x_file_parser_t *p,
                                   const uint8_t      *data,
                                   size_t              length,
                                   float               framerate)
{
    if (!p || !data || length < START_CODE_3LEN || framerate <= 0.0f) {
        return;
    }

    // Compute timestamp increment = round(90000 / framerate)
    double exact_inc = 90000.0 / framerate;
    p->timestamp_increment = (uint64_t)(exact_inc + 0.5);
    p->next_timestamp_90k  = 0;

    // Reset any prior state
    p->frame_len   = 0;
    p->in_frame    = false;
    memset(&p->nal_stats, 0, sizeof(p->nal_stats));

    size_t pos = 0;

    // 1) Find first start code (3- or 4-byte)
    {
        bool found = false;
        while (pos + START_CODE_3LEN <= length) {
            int sc_len = get_start_code_length(data, pos, length);
            if (sc_len > 0) {
                found = true;
                break;
            }
            pos++;
        }
        if (!found || pos + START_CODE_3LEN > length) {
            return; // no start code at all
        }
    }

    // Main loop: for each detected start code at 'pos'
    while (pos + START_CODE_3LEN <= length) {
        int sc_len = get_start_code_length(data, pos, length);
        if (sc_len == 0) {
            // This should not happen: we always enter the loop only when there's a start code.
            break;
        }

        size_t nal_payload_start = pos + sc_len;
        if (nal_payload_start >= length) {
            // no payload after this start code
            break;
        }

        // 2) Find next start code (3- or 4-byte) after nal_payload_start
        size_t next_pos = nal_payload_start;
        int    next_sc_len = 0;
        while (next_pos + START_CODE_3LEN <= length) {
            next_sc_len = get_start_code_length(data, next_pos, length);
            if (next_sc_len > 0) {
                break;
            }
            next_pos++;
        }

        // If we found next start code, nal_end = next_pos. Else nal_end = length.
        size_t nal_end = (next_sc_len > 0 && next_pos + START_CODE_3LEN <= length)
                         ? next_pos
                         : length;

        // Full NAL size = nal_end - pos
        int nal_size = (int)(nal_end - pos);

        // The “header byte” is at data[nal_payload_start]
        uint8_t nal_header = data[nal_payload_start];

        // Determine nal_type to see if it’s a VCL NAL (i.e. start of a new Access Unit)
        uint8_t nal_type = 0;
        if (p->codec == H26X_CODEC_H264) {
            nal_type = nal_header & 0x1F;
        } else {
            nal_type = (nal_header >> 1) & 0x3F;
        }

        // In H.264: VCL if nal_type in [1..5]
        // In H.265: VCL if nal_type in [0..31]
        bool is_vcl = false;
        if (p->codec == H26X_CODEC_H264) {
            if (nal_type >= 1 && nal_type <= 5) {
                is_vcl = true;
            }
        } else {
            if (nal_type <= 31) {
                is_vcl = true;
            }
        }

        if (is_vcl) {
            // If we were already in a frame, emit the previous one now:
            if (p->in_frame) {
                emit_parsed_frame(p);
            }
            // Start buffering a new Access Unit
            p->in_frame = true;
        }

        // If we are in a frame (or this is the first VCL), append the entire NAL (including its exact start code).
        if (p->in_frame) {
            append_nal_to_frame(p,
                                data + pos,   /* pointer to the start code itself */
                                nal_size,
                                nal_header);
        }

        // Advance pos to next_pos (if next_pos points at a start code) or break if end-of-buffer
        if (next_sc_len > 0 && next_pos + START_CODE_3LEN <= length) {
            pos = next_pos;
        } else {
            break;
        }
    }

    // After loop, if there’s a buffered frame, emit it now:
    if (p->in_frame && p->frame_len > 0) {
        emit_parsed_frame(p);
    }
}

/**
 * Create a new parser instance.
 */
h26x_file_parser_t *
h26x_file_parser_create(h26x_codec_t            codec,
                        void                   *context,
                        h26x_frame_callback_fn  cb)
{
    if (codec != H26X_CODEC_H264 && codec != H26X_CODEC_H265) {
        return NULL;
    }

    h26x_file_parser_t *p = (h26x_file_parser_t *)calloc(1, sizeof(*p));
    if (!p) {
        return NULL;
    }

    p->codec       = codec;
    p->context     = context;
    p->cb          = cb;
    p->frame_buffer = (uint8_t *)malloc(MAX_FRAME_SIZE);
    if (!p->frame_buffer) {
        free(p);
        return NULL;
    }

    // Initialize state
    p->frame_len           = 0;
    p->in_frame            = false;
    memset(&p->nal_stats, 0, sizeof(p->nal_stats));
    p->next_timestamp_90k  = 0;
    p->timestamp_increment = 0;
    return p;
}

/**
 * Destroy a parser, free its buffer.
 */
void
h26x_file_parser_destroy(h26x_file_parser_t *parser)
{
    if (!parser) return;
    free(parser->frame_buffer);
    free(parser);
}

/**
 * Reset the parser’s internal state so you can reuse it for another buffer.
 */
void
h26x_file_parser_reset(h26x_file_parser_t *parser)
{
    if (!parser) return;
    parser->frame_len           = 0;
    parser->in_frame            = false;
    parser->next_timestamp_90k  = 0;
    parser->timestamp_increment = 0;
    memset(&parser->nal_stats, 0, sizeof(parser->nal_stats));
}
