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

// Bounded search for a needle in a haystack buffer (like memmem)
static const uint8_t* buf_find(const uint8_t* hay, size_t hay_len,
                               const char* needle, size_t nee_len)
{
    if (nee_len == 0 || hay_len < nee_len) return NULL;
    // naive scan is fine for small SDP headers
    size_t last = hay_len - nee_len;
    for (size_t i = 0; i <= last; ++i) {
        if (hay[i] == (uint8_t)needle[0] &&
            memcmp(hay + i, needle, nee_len) == 0) {
            return hay + i;
        }
    }
    return NULL;
}

bool parse_pcap_sdp(const char *filename, char *sdp_buffer, size_t sdp_buffer_length) {
    char errbuf[PCAP_ERRBUF_SIZE] = {0};
    pcap_t *handle = pcap_open_offline_with_tstamp_precision(
        filename, PCAP_TSTAMP_PRECISION_MICRO, errbuf);
    if (!handle) {
        log_error("Could not open file %s", filename);
        return false;
    }

    struct pcap_pkthdr *hdr;
    const u_char *pkt;
    int res;

    // Constants we’ll search for (no strlen() at runtime)
    static const char CT_SDP[] = "Content-Type: application/sdp";
    static const char SDP_START[] = "v=0\r\n";
    static const char HDR_END[] = "\r\n\r\n";

    while ((res = pcap_next_ex(handle, &hdr, &pkt)) >= 0) {
        if (res == 0) continue; // timeout

        // Determine L2 offset (we’ll reject unknowns)
        int link_type = pcap_datalink(handle);
        size_t l2 = 0;
        switch (link_type) {
            case DLT_EN10MB:     l2 = 14; break; // Ethernet
            case DLT_LINUX_SLL:  l2 = 16; break; // Linux cooked
            case DLT_RAW:        l2 = 0;  break; // no L2
            default:
                log_error("Unsupported link type: %d", link_type);
                continue;
        }

        if (hdr->caplen < l2 + sizeof(struct ip)) continue; // not enough for IPv4 header
        const struct ip *ip = (const struct ip*)(pkt + l2);

        // Only handle IPv4/TCP here
        if (ip->ip_v != 4 || ip->ip_p != IPPROTO_TCP) continue;

        size_t ip_hl = (size_t)ip->ip_hl * 4;
        if (hdr->caplen < l2 + ip_hl + sizeof(struct tcphdr)) continue;

        const struct tcphdr *tcp = (const struct tcphdr*)((const uint8_t*)ip + ip_hl);
        size_t tcp_hl = (size_t)tcp->th_off * 4;

        if (hdr->caplen < l2 + ip_hl + tcp_hl) continue;

        // Compute payload pointer and lengths
        const uint8_t *payload = (const uint8_t*)tcp + tcp_hl;

        // Total IP payload length from header (network order)
        uint16_t ip_total_len = ntohs(ip->ip_len);
        if (ip_total_len < ip_hl + tcp_hl) continue; // malformed

        size_t payload_len_from_ip = (size_t)ip_total_len - ip_hl - tcp_hl;

        // Also bound by captured bytes
        size_t bytes_from_l2 = hdr->caplen - l2;
        if (bytes_from_l2 < ip_hl + tcp_hl) continue; // already checked, but be safe
        size_t payload_len_from_cap = bytes_from_l2 - ip_hl - tcp_hl;

        size_t payload_len = payload_len_from_ip;
        if (payload_len > payload_len_from_cap) {
            // Truncated capture; clamp to captured length
            payload_len = payload_len_from_cap;
        }
        if (payload_len == 0) continue;

        // Filter by port (bounded reads only)
        uint16_t sport = ntohs(tcp->th_sport);
        uint16_t dport = ntohs(tcp->th_dport);
        if (sport != SDP_TCP_PORT && dport != SDP_TCP_PORT) continue;

        // 1) Check for the content-type header within payload_len
        const uint8_t *ct = buf_find(payload, payload_len, CT_SDP, sizeof(CT_SDP) - 1);
        if (!ct) continue;

        // 2) Find start of SDP (v=0\r\n) after headers begin (search the whole payload safely)
        const uint8_t *sdp = buf_find(payload, payload_len, SDP_START, sizeof(SDP_START) - 1);
        if (!sdp) continue;

        // 3) Find end of headers/body separator: \r\n\r\n after sdp start (still bounded)
        const uint8_t *after_sdp = sdp; // search from sdp start (safe but conservative)
        size_t remaining = payload_len - (size_t)(after_sdp - payload);
        const uint8_t *end = buf_find(after_sdp, remaining, HDR_END, sizeof(HDR_END) - 1);

        // Compute SDP length (bounded); if no header terminator, take rest of payload
        size_t sdp_len;
        if (end && end >= sdp) {
            sdp_len = (size_t)(end - sdp);
        } else {
            sdp_len = payload_len - (size_t)(sdp - payload);
        }

        // Copy into caller’s buffer with NUL termination
        if (sdp_len + 1 <= sdp_buffer_length) {
            memcpy(sdp_buffer, sdp, sdp_len);
            sdp_buffer[sdp_len] = '\0';
            pcap_close(handle);
            return true;
        }
        // If too long, skip this packet and keep looking
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
