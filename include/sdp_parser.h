// sdp_parser.h

#ifndef __SDP_PARSER_H
#define __SDP_PARSER_H

#include <string>
#include <vector>
#include <optional>
#include <map>

typedef enum sdp_type {
    SDP_TYPE_NALU=0x01,
    SDP_TYPE_RTP=0x02
} sdp_type_t;

struct RtpCryptoInfo {
    std::string tag;
    std::string cryptoSuite;
    std::string keyParams;  // base64-decoded
};

struct RtpParameters {
    int port;
    int payloadType;
    std::string codec;
    uint32_t clockRate;               // RTP clock rate (e.g., 90000 for video, 48000 for OPUS)
    std::optional<std::string> sps;
    std::optional<std::string> pps;
    std::optional<std::string> vps;
    std::vector<RtpCryptoInfo> cryptoInfos;
};

// Parsed SDP holds both video and audio streams
struct parsed_sdp {
    std::vector<RtpParameters> videoStreams;
    std::vector<RtpParameters> audioStreams;
};
typedef struct parsed_sdp parsed_sdp_t;

parsed_sdp_t *parse_sdp(const char *sdp);
void parsed_sdp_destroy(parsed_sdp_t *p);
void print_parsed_sdp(const parsed_sdp_t *p);

#endif
