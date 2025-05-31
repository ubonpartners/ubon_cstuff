// rtp_receiver.c (Stage 1 - Reorder & Filter Buffer)

#include "rtp_receiver.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h> // ntohs, ntohl

#define MAX_REORDER_BUFFER 64
#define MAX_PACKET_SIZE 1500
#define RTP_VERSION 2
#define RTP_HEADER_MIN_SIZE 12
#define RTP_SSRC_SWITCH_THRESHOLD 4

typedef struct rtp_packet_slot {
    uint8_t buffer[MAX_PACKET_SIZE];
    int length;
    bool valid;
    uint16_t seq;
} rtp_packet_slot_t;

struct rtp_receiver {
    void *context;
    rtp_packet_callback_fn cb;

    uint32_t current_ssrc;
    bool ssrc_valid;
    uint32_t candidate_ssrc;
    int candidate_ssrc_count;

    int64_t last_extended_timestamp;
    uint32_t last_timestamp;
    uint16_t last_sequence;
    bool first_packet;

    uint8_t payload_type;
    bool payload_type_set;

    rtp_packet_slot_t buffer[MAX_REORDER_BUFFER];

    rtp_stats_t stats;
};

static inline int seq_num_diff(uint16_t a, uint16_t b) {
    return (int)((uint16_t)(a - b));
}

static inline int reorder_index(uint16_t seq) {
    return seq % MAX_REORDER_BUFFER;
}

static uint64_t extend_timestamp(uint32_t ts, rtp_receiver_t *r) {
    if (r->first_packet) {
        r->last_extended_timestamp = ts;
        r->last_timestamp = ts;
        return ts;
    }
    int32_t delta = (int32_t)(ts - r->last_timestamp);
    r->last_timestamp = ts;
    r->last_extended_timestamp += delta;
    return r->last_extended_timestamp;
}

void rtp_receiver_set_payload_type(rtp_receiver_t *r, uint8_t pt) {
    r->payload_type = pt;
    r->payload_type_set = true;
}

rtp_receiver_t *rtp_receiver_create(void *context, rtp_packet_callback_fn cb) {
    rtp_receiver_t *r = calloc(1, sizeof(rtp_receiver_t));
    r->context = context;
    r->cb = cb;
    return r;
}

void rtp_receiver_reset(rtp_receiver_t *r) {
    memset(r->buffer, 0, sizeof(r->buffer));
    memset(&r->stats, 0, sizeof(rtp_stats_t));
    r->first_packet = true;
    r->ssrc_valid = false;
    r->candidate_ssrc_count = 0;
}

void rtp_receiver_destroy(rtp_receiver_t *r) {
    free(r);
}

void rtp_receiver_fill_stats(rtp_receiver_t *r, rtp_stats_t *stats) {
    *stats = r->stats;
}

void rtp_receiver_add_packet(rtp_receiver_t *r, uint8_t *data, int length) {
    if (length < RTP_HEADER_MIN_SIZE || data[0] >> 6 != RTP_VERSION) {
        r->stats.packets_discarded_corrupt++;
        return;
    }

    uint8_t cc = data[0] & 0x0F;
    bool ext = !!(data[0] & 0x10);
    uint8_t pt = data[1] & 0x7F;
    bool marker = !!(data[1] & 0x80);
    uint16_t seq = ntohs(*(uint16_t *)(data + 2));
    uint32_t ts = ntohl(*(uint32_t *)(data + 4));
    uint32_t ssrc = ntohl(*(uint32_t *)(data + 8));

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

    if (r->payload_type_set && pt != r->payload_type) {
        r->stats.packets_discarded_wrong_pt++;
        return;
    }

    if (!r->ssrc_valid) {
        r->current_ssrc = ssrc;
        r->ssrc_valid = true;
    } else if (ssrc != r->current_ssrc) {
        if (ssrc == r->candidate_ssrc) {
            r->candidate_ssrc_count++;
            if (r->candidate_ssrc_count >= RTP_SSRC_SWITCH_THRESHOLD) {
                r->current_ssrc = ssrc;
                r->candidate_ssrc_count = 0;
            } else {
                r->stats.packets_discarded_wrong_ssrc++;
                return;
            }
        } else {
            r->candidate_ssrc = ssrc;
            r->candidate_ssrc_count = 1;
            r->stats.packets_discarded_wrong_ssrc++;
            return;
        }
    }

    int idx = reorder_index(seq);
    rtp_packet_slot_t *slot = &r->buffer[idx];

    if (slot->valid && slot->seq == seq) {
        r->stats.packets_duplicated++;
        return;
    }

    memcpy(slot->buffer, data, length);
    slot->length = length;
    slot->seq = seq;
    slot->valid = true;
    r->stats.packets_received++;

    // Try to output in-sequence packets
    if (r->first_packet) {
        r->last_sequence = seq - 1;
        r->first_packet = false;
    }

    while (1) {
        uint16_t next_seq = r->last_sequence + 1;
        int out_idx = reorder_index(next_seq);
        rtp_packet_slot_t *out_slot = &r->buffer[out_idx];
        if (!out_slot->valid || out_slot->seq != next_seq) {
            break;
        }

        uint8_t *out_data = out_slot->buffer;
        uint8_t out_cc = out_data[0] & 0x0F;
        bool out_ext = !!(out_data[0] & 0x10);

        int out_header_len = RTP_HEADER_MIN_SIZE + out_cc * 4;
        if (out_ext) {
            uint16_t ext_len_words = ntohs(*(uint16_t *)(out_data + out_header_len + 2));
            out_header_len += 4 + ext_len_words * 4;
        }

        rtp_packet_t pkt = {
            .data = out_data,
            .total_length = out_slot->length,
            .payload_offset = out_header_len,
            .payload_length = out_slot->length - out_header_len,
            .sequence_number = next_seq,
            .timestamp = ntohl(*(uint32_t *)(out_data + 4)),
            .extended_timestamp_90khz = extend_timestamp(ntohl(*(uint32_t *)(out_data + 4)), r),
            .ssrc = r->current_ssrc,
            .marker = !!(out_data[1] & 0x80)
        };

        r->cb(r->context, &pkt);
        out_slot->valid = false;
        r->last_sequence = next_seq;
    }
}