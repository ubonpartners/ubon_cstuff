#include "rtp_receiver.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <arpa/inet.h>     // for ntohs, ntohl
#include <srtp2/srtp.h>    // libsrtp2 main header

// Maximum size of our circular reorder buffer (power‐of‐two is easiest but 32 works).
#define MAX_REORDER_BUFFER   32
#define MAX_PACKET_SIZE      1500
#define RTP_VERSION          2
#define RTP_HEADER_MIN_SIZE  12
#define RTP_SSRC_SWITCH_THRESHOLD  4

/**
 * Each slot holds one RTP packet (raw bytes), plus metadata:
 *   - 'seq' is the 16‐bit sequence number
 *   - 'timestamp32' is the 32‐bit RTP timestamp from the header
 *   - 'valid' indicates whether the slot currently holds an un‐output packet
 */
typedef struct rtp_packet_slot {
    uint8_t  buffer[MAX_PACKET_SIZE];
    int      length;
    bool     valid;
    uint16_t seq;
    uint32_t timestamp32;
} rtp_packet_slot_t;

/*
 * The main receiver struct:
 *   - context, callback
 *   - SSRC‐handling fields
 *   - RTP timestamp extension logic
 *   - reorder buffer
 *   - “last output” state (last_sequence, last timestamps)
 *   - newly‐added: max_delay_packets, max_delay_ms
 *   - stats (including packets_late)
 *   - SRTP state (session handle, enabled flag)
 */
struct rtp_receiver {
    void *context;
    rtp_packet_callback_fn cb;

    uint32_t current_ssrc;
    bool     ssrc_valid;
    uint32_t candidate_ssrc;
    int      candidate_ssrc_count;

    // For 90 kHz extended‐timestamp bookkeeping
    int64_t  last_extended_timestamp_90khz;
    uint32_t last_timestamp32;       // last RTP timestamp (32‐bit) that we actually output
    bool     first_packet;

    uint8_t  payload_type;
    bool     payload_type_set;

    // Reorder buffer array (circular by seq % MAX_REORDER_BUFFER)
    rtp_packet_slot_t buffer[MAX_REORDER_BUFFER];

    // New timeout parameters (in pkt‐count or ms)
    uint32_t max_delay_packets;
    uint32_t max_delay_ms;

    // The last sequence number we handed off to the user
    uint16_t last_sequence;

    // count packets that have totally wrong sequence number
    // if too many in a row, reset
    uint32_t sequential_packets_outside_window;

    // Stats, including new field for late arrivals
    rtp_stats_t stats;

    // --- SRTP fields ---
    srtp_t   srtp_session;
    bool     srtp_enabled;
};

/*
 * Helper: signed difference of two 16‐bit seq numbers, wrapping at 0x10000.
 *   seq_num_diff(a,b) = (int16_t)(a − b)
 *   If result > 0, a is “ahead” of b. If < 0, a is behind b. If = 0, equal.
 */
static inline int seq_num_diff(uint16_t a, uint16_t b) {
    return (int16_t)(a - b);
}

/*
 * Given a 16‐bit seq, find its index in the reorder buffer:
 *   idx = seq mod MAX_REORDER_BUFFER
 */
static inline int reorder_index(uint16_t seq) {
    return (int)(seq % MAX_REORDER_BUFFER);
}

/*
 * Whenever we output a packet, we call extend_timestamp(…).
 *   - On the very first packet, just set last_extended = raw ts.
 *   - Later, delta = (int32_t)(current_32bit_ts − last_32bit_ts),
 *     and we add that delta (signed) to last_extended to get a 64‐bit extended timestamp.
 */
static uint64_t extend_timestamp(uint32_t ts32, rtp_receiver_t *r) {
    if (r->first_packet) {
        r->last_extended_timestamp_90khz = ts32;
        r->last_timestamp32 = ts32;
        return (uint64_t)ts32;
    }
    int32_t delta = (int32_t)(ts32 - r->last_timestamp32);
    r->last_timestamp32 = ts32;
    r->last_extended_timestamp_90khz += (int64_t)delta;
    return (uint64_t)r->last_extended_timestamp_90khz;
}

/*
 * Public API: set payload type filter (optional).
 */
void rtp_receiver_set_payload_type(rtp_receiver_t *r, uint8_t pt) {
    r->payload_type = pt;
    r->payload_type_set = true;
}

/*
 * Public API: set the reorder‐timeout parameters.
 *   max_delay_packets: once this many packets “beyond” a missing seq are in the buffer, skip it
 *   max_delay_ms: once any buffered packet’s RTP‐timestamp is more than 90*max_delay_ms beyond last_output, skip
 */
void rtp_receiver_set_max_delay(rtp_receiver_t *r,
                                uint32_t max_delay_packets,
                                uint32_t max_delay_ms)
{
    r->max_delay_packets = max_delay_packets;
    r->max_delay_ms      = max_delay_ms;
}

/*
 * Create a new receiver.  Zero‐initialize everything (buffer slots, stats, etc.).
 * Also calls srtp_init() so that any subsequent SRTP session creation will succeed.
 */
rtp_receiver_t *rtp_receiver_create(void *context, rtp_packet_callback_fn cb) {
    rtp_receiver_t *r = (rtp_receiver_t *)calloc(1, sizeof(rtp_receiver_t));
    if (!r) return NULL;

    // One-time libsrtp initialization
    srtp_init();

    r->context      = context;
    r->cb           = cb;
    r->first_packet = true;

    // Defaults: no payload_type set, no SSRC, timeouts = 0
    r->payload_type_set     = false;
    r->ssrc_valid           = false;
    r->candidate_ssrc_count = 0;
    r->max_delay_packets    = 4;
    r->max_delay_ms         = 100;
    r->last_sequence        = 0;  // will be set upon first packet
    memset(&r->stats, 0, sizeof(rtp_stats_t));

    // By default, SRTP is disabled
    r->srtp_enabled = false;
    r->srtp_session = NULL;

    // Buffer slots already zeroed by calloc (buffer[].valid == false)
    return r;
}

/*
 * Reset to “no packets seen.”  Clears buffer, stats, SSRC, etc.
 * Does NOT tear down SRTP; you can continue using the same SRTP session.
 */
void rtp_receiver_reset(rtp_receiver_t *r) {
    memset(r->buffer, 0, sizeof(r->buffer));
    memset(&r->stats, 0, sizeof(rtp_stats_t));
    r->first_packet            = true;
    r->ssrc_valid              = false;
    r->candidate_ssrc_count    = 0;
    r->payload_type_set        = false;
    r->last_sequence           = 0;
    r->last_extended_timestamp_90khz = 0;
    r->last_timestamp32        = 0;
    r->sequential_packets_outside_window = 0;
    // timeouts stay as they were (since presumably user set them)
}

/*
 * Destroy/free the receiver.
 * If SRTP was enabled, deallocate the SRTP session.
 */
void rtp_receiver_destroy(rtp_receiver_t *r) {
    if (!r) return;

    if (r->srtp_enabled && r->srtp_session != NULL) {
        srtp_dealloc(r->srtp_session);
    }
    free(r);
}

/*
 * Copy out the stats.
 */
void rtp_receiver_fill_stats(rtp_receiver_t *r, rtp_stats_t *stats) {
    *stats = r->stats;
}

/*
 * Enable SRTP on an existing receiver.  'key' must point to the SRTP master key bytes,
 * and 'key_len' its length.  Returns 0 on success, or -1 on failure.
 *
 * Example: for AES_CM_128_HMAC_SHA1_80, use a 30‐byte key (16‐byte AES key + 14‐byte salt).
 */
int rtp_receiver_enable_srtp(rtp_receiver_t *r, const uint8_t *key, size_t key_len) {
    if (!r || !key) return -1;

    srtp_policy_t policy;
    srtp_err_status_t err;

    // Use the common crypto policy: AES_CM_128_HMAC_SHA1_80
    srtp_crypto_policy_set_aes_cm_128_hmac_sha1_80(&policy.rtp);
    srtp_crypto_policy_set_aes_cm_128_hmac_sha1_80(&policy.rtcp);

    // Accept any inbound SSRC
    policy.ssrc.type  = ssrc_any_inbound;
    policy.ssrc.value = 0;
    policy.key        = (uint8_t *)key;
    policy.next       = NULL;

    err = srtp_create(&r->srtp_session, &policy);
    if (err != srtp_err_status_ok) {
        return -1;
    }
    r->srtp_enabled = true;
    return 0;
}

/*
 * This is the main function.  We parse the RTP header, check SSRC/PT if needed,
 * insert into the circular buffer, then try flushing all in‐order packets.
 *
 * We also check, at the very top, for “late” packets (seq ≤ last_sequence).  Those
 * are either duplicates or true late arrivals (we have already skipped their slot).
 *
 * With SRTP enabled, we first call srtp_unprotect() to decrypt & authenticate.
 */
void rtp_receiver_add_packet(rtp_receiver_t *r, uint8_t *data, int length) {
    // If SRTP is enabled, decrypt/authenticate first.
    if (r->srtp_enabled) {
        int unprotect_len = length;
        srtp_err_status_t serr = srtp_unprotect(r->srtp_session, data, &unprotect_len);
        if (serr != srtp_err_status_ok) {
            r->stats.packets_discarded_corrupt++;
            return;
        }
        length = unprotect_len;
        if (length < RTP_HEADER_MIN_SIZE) {
            // Too small even after stripping auth tag
            r->stats.packets_discarded_corrupt++;
            return;
        }
    }

    // 1) Quick validity checks on RTP header
    if (length < RTP_HEADER_MIN_SIZE || (data[0] >> 6) != RTP_VERSION) {
        r->stats.packets_discarded_corrupt++;
        return;
    }
    uint8_t  cc      = data[0] & 0x0F;
    bool     ext     = !!(data[0] & 0x10);
    uint8_t  pt      = data[1] & 0x7F;
    bool     marker  = !!(data[1] & 0x80);
    uint16_t seq     = ntohs(*(uint16_t *)(data + 2));
    uint32_t ts32    = ntohl(*(uint32_t *)(data + 4));
    uint32_t ssrc    = ntohl(*(uint32_t *)(data + 8));

    int header_len = RTP_HEADER_MIN_SIZE + cc * 4;
    if (ext) {
        if (length < header_len + 4) {
            r->stats.packets_discarded_corrupt++;
            return;
        }
        uint16_t ext_len_words = ntohs(*(uint16_t *)(data + header_len + 2));
        header_len += 4 + ext_len_words * 4;
    }
    if (header_len > length || header_len >= MAX_PACKET_SIZE) {
        r->stats.packets_discarded_corrupt++;
        return;
    }

    // 2) Payload‐type filter, if set
    if (r->payload_type_set && pt != r->payload_type) {
        r->stats.packets_discarded_wrong_pt++;
        return;
    }

    // 3) SSRC handling / switching
    if (!r->ssrc_valid) {
        r->current_ssrc = ssrc;
        r->ssrc_valid   = true;
    } else if (ssrc != r->current_ssrc) {
        if (ssrc == r->candidate_ssrc) {
            r->candidate_ssrc_count++;
            if (r->candidate_ssrc_count >= RTP_SSRC_SWITCH_THRESHOLD) {
                // Switch
                r->current_ssrc = ssrc;
                r->candidate_ssrc_count = 0;
            } else {
                r->stats.packets_discarded_wrong_ssrc++;
                return;
            }
        } else {
            r->candidate_ssrc       = ssrc;
            r->candidate_ssrc_count = 1;
            r->stats.packets_discarded_wrong_ssrc++;
            return;
        }
    }

    // 4) If we have already output at least one packet, check for “late” or “duplicate”
    if (!r->first_packet) {
        int diff = seq_num_diff(seq, r->last_sequence);
        if ((diff < -MAX_REORDER_BUFFER) || (diff > MAX_REORDER_BUFFER)) {
            // this is bad :(
            r->stats.packets_outside_window++;
            r->sequential_packets_outside_window++;
            if (r->sequential_packets_outside_window > 4) {
                r->last_sequence = (seq - 1) & 0xFFFF;
                r->stats.resets++;
            }
        } else {
            r->sequential_packets_outside_window = 0;
        }

        if (diff < 0) {
            r->stats.packets_late++;
            return;
        }
    }

    // 5) Insert into circular reorder buffer
    {
        int idx = reorder_index(seq);
        rtp_packet_slot_t *slot = &r->buffer[idx];

        // If this slot already holds exactly the same seq, it's a duplicate
        if (slot->valid && slot->seq == seq) {
            r->stats.packets_duplicated++;
            // we'll take this copy
        }

        // Otherwise, copy data in
        memcpy(slot->buffer, data, length);
        slot->length      = length;
        slot->valid       = true;
        slot->seq         = seq;
        slot->timestamp32 = ts32;   // record the raw 32‐bit RTP timestamp
        r->stats.packets_received++;
    }

    // 6) If this is the very first packet we see, initialize last_sequence so that
    //    the “next expected” is exactly this seq (i.e. last_sequence = seq − 1).
    if (r->first_packet) {
        r->last_sequence = seq - 1;
        r->first_packet  = false;
    }

    // 7) Now try to flush any in‐order packets, possibly skipping holes if timeouts/fire.
    while (1) {
        uint16_t next_seq = r->last_sequence + 1;
        int      out_idx  = reorder_index(next_seq);
        rtp_packet_slot_t *out_slot = &r->buffer[out_idx];

        // If the exact next‐seq slot is present and valid, output it:
        if (out_slot->valid && out_slot->seq == next_seq) {
            // parse header to find payload offset/length
            uint8_t *out_data = out_slot->buffer;
            uint8_t  out_cc   = out_data[0] & 0x0F;
            bool     out_ext  = !!(out_data[0] & 0x10);

            int out_header_len = RTP_HEADER_MIN_SIZE + out_cc * 4;
            if (out_ext) {
                uint16_t ext_len_words = ntohs(*(uint16_t *)(out_data + out_header_len + 2));
                out_header_len += 4 + ext_len_words * 4;
            }

            // Build rtp_packet_t and call callback
            rtp_packet_t pkt;
            pkt.data                    = out_data;
            pkt.total_length            = out_slot->length;
            pkt.payload_offset          = out_header_len;
            pkt.payload_length          = out_slot->length - out_header_len;
            pkt.sequence_number         = next_seq;
            pkt.timestamp               = out_slot->timestamp32;
            pkt.extended_timestamp_90khz = extend_timestamp(out_slot->timestamp32, r);
            pkt.ssrc                    = r->current_ssrc;
            pkt.marker                  = !!(out_data[1] & 0x80);

            // Remember this as “last output” so that future timestamp comparisons can use it
            r->last_sequence = next_seq;

            // Invoke user callback
            r->cb(r->context, &pkt);

            // Remove from buffer
            out_slot->valid = false;
            // (r->last_timestamp32 and r->last_extended_timestamp_90khz were updated in extend_timestamp)

            // Continue looping to see if the next‐next sequence is already here
            continue;
        }

        // If we reach here, the “next_seq” slot is not present (or seq mismatch), so maybe we skip it.
        bool skip_missing = false;

        // A) Packet‐count condition:
        if (r->max_delay_packets > 0) {
            int lookahead_count = 0;
            // Scan up to MAX_REORDER_BUFFER slots “ahead.” If we count ≥ max_delay_packets
            // valid packets whose sequence is strictly > next_seq, then we skip “next_seq.”
            for (int offset = 1; offset < MAX_REORDER_BUFFER; offset++) {
                int future_seq = (next_seq + offset) & 0xFFFF;
                int idx2 = reorder_index(future_seq);
                rtp_packet_slot_t *s2 = &r->buffer[idx2];
                if (s2->valid && future_seq == s2->seq) {
                    // Only count packets that are truly ahead of next_seq
                    lookahead_count++;
                    if ((uint32_t)lookahead_count >= r->max_delay_packets) {
                        skip_missing = true;
                        break;
                    }
                    int32_t delta32 = (int32_t)(s2->timestamp32 - r->last_timestamp32);
                    if ((int64_t)delta32 > (int64_t)(90 * r->max_delay_ms)) {
                        skip_missing = true;
                        break;
                    }
                }
            }
        }

        if (skip_missing) {
            // We skip “next_seq” even though it was never received.
            r->last_sequence = next_seq;
            r->stats.packets_missing += 1;
            // Note: do NOT touch r->last_timestamp32 or r->last_extended_timestamp_90khz
            continue;
        }

        // No skip condition met, and no actual packet to output → we stop for now.
        break;
    }
}

void print_rtp_stats(const rtp_stats_t *stats) {
    printf("RTP Stats:\n");
    printf("  Received:           %u\n", stats->packets_received);
    printf("  Missing:            %u\n", stats->packets_missing);
    printf("  Duplicated:         %u\n", stats->packets_duplicated);
    printf("  Discarded (corrupt):%u\n", stats->packets_discarded_corrupt);
    printf("  Discarded (SSRC):   %u\n", stats->packets_discarded_wrong_ssrc);
    printf("  Discarded (PT):     %u\n", stats->packets_discarded_wrong_pt);
    printf("  Late arrivals:      %u\n", stats->packets_late);
    printf("  Outside window:     %u\n", stats->packets_outside_window);
    printf("  Resets:             %u\n", stats->resets);
}
