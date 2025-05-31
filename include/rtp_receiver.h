// rtp_receiver.h
#ifndef RTP_RECEIVER_H
#define RTP_RECEIVER_H

#include <stdint.h>
#include <stdbool.h>

/*
 * User supplies a callback of this type to receive each in‐order RTP packet.
 *   context: whatever the user passed into rtp_receiver_create()
 *   pkt: full metadata + pointers into data
 */
typedef struct rtp_packet {
    const uint8_t *data;               // Pointer to full RTP packet buffer
    int            total_length;       // Full length of the RTP packet
    int            payload_offset;     // Offset into 'data' where payload begins
    int            payload_length;     // Length of the payload
    uint16_t       sequence_number;    // 16‐bit RTP sequence number
    uint32_t       timestamp;          // 32‐bit RTP timestamp (raw)
    uint64_t       extended_timestamp_90khz;  // “Extended” RTP timestamp in 90 kHz domain
    uint32_t       ssrc;               // SSRC of the stream
    bool           marker;             // Marker bit from header
} rtp_packet_t;

typedef void (*rtp_packet_callback_fn)(void *context, const rtp_packet_t *pkt);

typedef struct rtp_stats {
    uint32_t packets_received;
    uint32_t packets_duplicated;
    uint32_t packets_discarded_corrupt;
    uint32_t packets_discarded_wrong_ssrc;
    uint32_t packets_discarded_wrong_pt;
    uint32_t packets_late;      // number of packets that arrived “after” we skipped them
} rtp_stats_t;

typedef struct rtp_receiver rtp_receiver_t;

/*
 * Create a new RTP receiver. 'context' is user‐supplied; 'cb' is called for each in‐order packet.
 */
rtp_receiver_t *rtp_receiver_create(void *context, rtp_packet_callback_fn cb);

/*
 * Reset the receiver to “no packets seen” (clears buffer & stats, SSRC, etc.)
 */
void rtp_receiver_reset(rtp_receiver_t *r);

/*
 * Free/destroy the receiver.
 */
void rtp_receiver_destroy(rtp_receiver_t *r);

/*
 * If you want to only accept one payload type, call this before sending packets.
 */
void rtp_receiver_set_payload_type(rtp_receiver_t *r, uint8_t pt);

/*
 * Configure the reorder buffer timeout parameters:
 *   max_delay_packets:  number of packets “ahead” that accumulate before we skip a missing one
 *   max_delay_ms:       how many milliseconds (RTP timestamp domain) before we skip
 *
 * Both defaults to zero (disabled) unless you set them explicitly.
 */
void rtp_receiver_set_max_delay(rtp_receiver_t *r,
                                uint32_t max_delay_packets,
                                uint32_t max_delay_ms);

/*
 * Supplies one full RTP packet (raw‐byte buffer + length).  The receiver will
 * attempt to emit in‐order packets via the callback.  Late packets (seq ≤ already‐output)
 * get dropped (and counted as r->stats.packets_late).
 */
void rtp_receiver_add_packet(rtp_receiver_t *r, uint8_t *data, int length);

/*
 * At any time, you can query statistics about what happened.
 */
void rtp_receiver_fill_stats(rtp_receiver_t *r, rtp_stats_t *stats);

#endif // RTP_RECEIVER_H
