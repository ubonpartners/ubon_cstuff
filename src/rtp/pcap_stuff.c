#include <pcap.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include "pcap_stuff.h"
#include "log.h"

#define SDP_TCP_PORT 554
#define RTP_UDP_PORT 5004

bool parse_pcap_sdp(const char *filename, char *sdp_buffer, size_t sdp_buffer_length) {
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t *handle = pcap_open_offline_with_tstamp_precision(filename, PCAP_TSTAMP_PRECISION_MICRO, errbuf);
    if (!handle)
    {
        log_error("Could not open file %s",filename);
        return false;
    }

    struct pcap_pkthdr *header;
    const u_char *packet;
    int res;

    while ((res = pcap_next_ex(handle, &header, &packet)) >= 0) {
        if (res == 0) continue;
        // Skip Ethernet + IP + TCP headers manually

        int link_type = pcap_datalink(handle);
        int l2_offset = 0;

        switch (link_type) {
            case DLT_EN10MB: // Ethernet
                l2_offset = 14;
                break;
            case DLT_LINUX_SLL: // Linux Cooked Capture
                l2_offset = 16;
                break;
            case DLT_RAW: // No L2 header
                l2_offset = 0;
                break;
            default:
                log_error("Unsupported link type: %d\n", link_type);
                continue;
        }

        const struct ip *ip_hdr = (struct ip *)(packet + l2_offset); // Ethernet header = 14 bytes
        if (ip_hdr->ip_p != IPPROTO_TCP) continue;

        size_t ip_hdr_len = ip_hdr->ip_hl * 4;
        const struct tcphdr *tcp_hdr = (const struct tcphdr *)((const uint8_t *)ip_hdr + ip_hdr_len);

        if (ntohs(tcp_hdr->th_sport) != SDP_TCP_PORT && ntohs(tcp_hdr->th_dport) != SDP_TCP_PORT)
            continue;

        size_t tcp_hdr_len = tcp_hdr->th_off * 4;
        const uint8_t *payload = (const uint8_t *)tcp_hdr + tcp_hdr_len;
        size_t payload_len = ntohs(ip_hdr->ip_len) - ip_hdr_len - tcp_hdr_len;

        if (payload_len > 0 && strstr((const char *)payload, "Content-Type: application/sdp")) {
            const char *sdp_start = strstr((const char *)payload, "v=0\r\n");
            if (sdp_start) {
                const char *end = strstr(sdp_start, "\r\n\r\n");
                size_t sdp_len = end ? (end - sdp_start) : strlen(sdp_start);
                if (sdp_len < sdp_buffer_length) {
                    memcpy(sdp_buffer, sdp_start, sdp_len);
                    sdp_buffer[sdp_len] = '\0';
                    pcap_close(handle);
                    return true;
                }
            }
        }
    }

    pcap_close(handle);
    return false;
}

void parse_pcap_rtp(const char *filename, void *callback_context, rtp_callback_fn rtp_callback) {
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t *handle = pcap_open_offline_with_tstamp_precision(filename, PCAP_TSTAMP_PRECISION_MICRO, errbuf);
    if (!handle) return;

    struct pcap_pkthdr *header;
    const u_char *packet;
    int res;

    while ((res = pcap_next_ex(handle, &header, &packet)) >= 0) {
        if (res == 0) continue;

        const struct ip *ip_hdr = (struct ip *)(packet + 14);
        if (ip_hdr->ip_p != IPPROTO_UDP) continue;

        size_t ip_hdr_len = ip_hdr->ip_hl * 4;
        const struct udphdr *udp_hdr = (const struct udphdr *)((const uint8_t *)ip_hdr + ip_hdr_len);

        size_t udp_payload_len = ntohs(udp_hdr->uh_ulen) - sizeof(struct udphdr);
        const uint8_t *udp_payload = (const uint8_t *)udp_hdr + sizeof(struct udphdr);

        if (udp_payload_len >= 12 && ((udp_payload[0] & 0xC0) == 0x80)) {
            double capture_time = header->ts.tv_sec + header->ts.tv_usec / 1e6;
            rtp_callback(callback_context, (uint8_t *)udp_payload, udp_payload_len, capture_time);
        }
    }

    pcap_close(handle);
}
