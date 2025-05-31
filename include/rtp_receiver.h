#ifndef __RTP_RECEIVER_H
#define __RTP_RECEIVER_H

#include <stdint.h>
#include <stdbool.h>

typedef struct rtp_receiver rtp_receiver_t;

typedef struct rtp_packet {
    const uint8_t *data;         // Pointer to full RTP packet buffer
    int total_length;            // Full length of RTP packet
    int payload_offset;          // Offset to RTP payload
    int payload_length;          // Length of payload
    uint16_t sequence_number;    // 16-bit RTP sequence number
    uint32_t timestamp;          // 32-bit RTP timestamp
    uint64_t extended_timestamp_90khz; // Extended RTP timestamp in 90kHz domain
    uint32_t ssrc;               // SSRC of the stream
    bool marker;                 // Marker bit
} rtp_packet_t;

typedef struct rtp_stats {
    uint32_t packets_received;
    uint32_t packets_duplicated;
    uint32_t packets_discarded_corrupt;
    uint32_t packets_discarded_wrong_ssrc;
    uint32_t packets_discarded_wrong_pt;
    // You may add more stats if needed
} rtp_stats_t;

typedef void (*rtp_packet_callback_fn)(void *context, const rtp_packet_t *packet);

rtp_receiver_t *rtp_receiver_create(void *context, rtp_packet_callback_fn cb);
void rtp_receiver_destroy(rtp_receiver_t *r);

void rtp_receiver_reset(rtp_receiver_t *r);
void rtp_receiver_set_payload_type(rtp_receiver_t *r, uint8_t pt); // Optional filter
void rtp_receiver_fill_stats(rtp_receiver_t *r, rtp_stats_t *stats);

// Add RTP packet (raw network byte buffer)
void rtp_receiver_add_packet(rtp_receiver_t *r, uint8_t *rtp_packet, int rtp_packet_length);


#endif // __RTP_RECEIVER_H