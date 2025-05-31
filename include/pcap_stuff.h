#ifndef __PCAP_STUFF_H
#define __PCAP_STUFF_H

#include <stdint.h>

typedef void (*rtp_callback_fn)(void *context,
                                uint8_t *rtp_packet, size_t packet_length, double capture_time);

bool parse_pcap_sdp(const char *filename, char *sdp_buffer, size_t sdp_buffer_length);
void parse_pcap_rtp(const char *filename, void *callback_context, rtp_callback_fn rtp_callback);


#endif
