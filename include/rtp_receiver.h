// rtp_receiver.h

#ifndef RTP_RECEIVER_H
#define RTP_RECEIVER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

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
    uint32_t packets_missing;
    uint32_t packets_duplicated;
    uint32_t packets_discarded_corrupt;
    uint32_t packets_discarded_wrong_ssrc;
    uint32_t packets_discarded_wrong_pt;
    uint32_t packets_late;      // number of packets that arrived “after” we skipped them
    uint32_t packets_outside_window;
    uint32_t resets;            // if seq is more than the reorder window out, triggers a reset
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
 * Both default to zero (disabled) unless you set them explicitly.
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

/*
 * Debug: quickly print the stats
 */
void print_rtp_stats(const rtp_stats_t *stats);

/*
 * --- SRTP SUPPORT ---
 *
 * Call this *after* creating the receiver, but *before* feeding packets.
 * 'key' is the SRTP master key (e.g. 30 bytes for AES_CM_128_HMAC_SHA1_80).
 * Returns 0 on success, or -1 on failure.
 */
int rtp_receiver_enable_srtp(rtp_receiver_t *r, const uint8_t *key, size_t key_len);


/**************************************************************************************************
 * New‐in‐this‐version: SDP parsing & UDP‐receive loop
 **************************************************************************************************/

/*
 * Parse a minimal SDP (Session Description Protocol) string and configure:
 *   - payload type filter (from the "m=... <port> RTP/AVP <fmt>" line)
 *   - SRTP key, if an "a=crypto:… inline:<base64key>" attribute is present
 *
 * Example SDP snippet:
 *   v=0
 *   o=- 0 0 IN IP4 127.0.0.1
 *   s=Example
 *   c=IN IP4 0.0.0.0
 *   t=0 0
 *   m=audio 5004 RTP/AVP 96
 *   a=rtpmap:96 opus/48000/2
 *   a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:MTIzNDU2Nzg5MDEyMzQ1Njc4OTAxMjM0NTY3ODkw  // base64-encoded key
 *
 * On success, returns 0.  On failure (e.g. malformed base64, no "m=" line), returns -1.
 *
 * After calling this, the receiver’s payload type is set automatically.  If an SRTP
 * inline key is found and successfully base64-decoded, SRTP is enabled under the hood.
 */
int rtp_receiver_set_sdp(rtp_receiver_t *r, const char *sdp_str);

/*
 * Start a background UDP‐receive thread on 'local_ip':'port'.  Every incoming UDP datagram
 * is immediately pushed into rtp_receiver_add_packet(...).  Returns 0 on success, or -1 on error.
 *
 * The caller can later stop it by calling rtp_receiver_stop_receive(r).
 *
 * NOTE: you must link with -lpthread for this to compile.
 */
int rtp_receiver_start_receive(rtp_receiver_t *r,
                               const char     *local_ip,
                               uint16_t        port);

/*
 * Stop the background UDP receiver (if running).  After this returns, no further
 * rtp_receiver_add_packet calls will happen.  If the thread was blocked in recvfrom(),
 * it will be unblocked and joined.
 */
void rtp_receiver_stop_receive(rtp_receiver_t *r);

#endif // RTP_RECEIVER_H
