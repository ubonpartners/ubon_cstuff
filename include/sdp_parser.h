#ifndef __SDP_PARSER_H
#define __SDP_PARSER_H

// sdp_parser.h
#include <string>
#include <vector>
#include <optional>
#include <map>

struct RtpCryptoInfo {
    std::string tag;
    std::string cryptoSuite;
    std::string keyParams;  // base64-decoded
};

struct RtpParameters {
    int port;
    int payloadType;
    std::string codec;
    std::optional<std::string> sps;
    std::optional<std::string> pps;
    std::optional<std::string> vps;
    std::vector<RtpCryptoInfo> cryptoInfos;
};

typedef struct parsed_sdp {
    std::vector<RtpParameters> videoStreams;
} parsed_sdp_t;

parsed_sdp_t *parse_sdp(const char *sdp);
void parsed_sdp_destroy(parsed_sdp_t *p);
void print_parsed_sdp(const parsed_sdp_t *p);


#endif
