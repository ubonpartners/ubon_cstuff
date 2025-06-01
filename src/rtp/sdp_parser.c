// sdp_parser.cpp
#include "sdp_parser.h"
#include <sstream>
#include <regex>
#include <cstring>
#include <iterator>
#include <iostream>
#include <cstdlib>
#include <cctype>
#include <memory>
#include <algorithm>
#include <vector>

static std::string base64_decode(const std::string &in) {
    std::string out;
    std::string b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[b64[i]] = i;

    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

parsed_sdp_t *parse_sdp(const char *sdp_text) {
    auto *parsed = new parsed_sdp_t;
    std::istringstream sdp(sdp_text);
    std::string line;

    RtpParameters current;
    bool in_video = false;

    while (std::getline(sdp, line)) {
        if (line.back() == '\r') line.pop_back();

        if (line.rfind("m=video", 0) == 0) {
            if (in_video)
                parsed->videoStreams.push_back(current);

            current = RtpParameters();
            in_video = true;

            std::istringstream iss(line.substr(2));  // skip "m=" prefix
            std::string media, proto;
            int port, pt;
            iss >> media >> port >> proto >> pt;

            current.port = port;
            current.payloadType = pt;
        } else if (in_video && line.rfind("a=rtpmap:", 0) == 0) {
            std::regex rtpmap_regex(R"a(a=rtpmap:(\d+)\s+([^\s/]+))a");
            std::smatch match;
            if (std::regex_search(line, match, rtpmap_regex)) {
                int pt = std::atoi(match[1].str().c_str());
                if (pt == current.payloadType)
                    current.codec = match[2];
            }
        } else if (in_video && line.rfind("a=crypto:", 0) == 0) {
            std::regex crypto_regex(R"(a=crypto:(\d+)\s+(\S+)\s+inline:([a-zA-Z0-9+/=]+))");
            std::smatch match;
            if (std::regex_search(line, match, crypto_regex)) {
                RtpCryptoInfo ci;
                ci.tag = match[1];
                ci.cryptoSuite = match[2];
                ci.keyParams = base64_decode(match[3]);
                current.cryptoInfos.push_back(ci);
            }
        } else if (in_video && line.rfind("a=fmtp:", 0) == 0) {
            std::regex sps_regex(R"(sprop-parameter-sets=([^,]+),([^;\r\n]+))");
            std::regex vps_regex(R"(sprop-vps=([^;\r\n]+))");
            std::smatch match;
            if (std::regex_search(line, match, sps_regex)) {
                current.sps = base64_decode(match[1]);
                current.pps = base64_decode(match[2]);
            }
            if (std::regex_search(line, match, vps_regex)) {
                current.vps = base64_decode(match[1]);
            }
        }
    }

    if (in_video)
        parsed->videoStreams.push_back(current);

    return parsed;
}

void parsed_sdp_destroy(parsed_sdp_t *p) {
    delete p;
}

void print_parsed_sdp(const parsed_sdp_t *parsed) {
    if (!parsed) {
        std::cout << "No SDP parsed.\n";
        return;
    }

    for (size_t i = 0; i < parsed->videoStreams.size(); ++i) {
        const auto &vs = parsed->videoStreams[i];
        std::cout << "Video Stream #" << i + 1 << ":\n";
        std::cout << "  Port: " << vs.port << "\n";
        std::cout << "  Payload Type: " << vs.payloadType << "\n";
        std::cout << "  Codec: " << vs.codec << "\n";

        if (vs.sps) {
            std::cout << "  SPS (decoded, " << vs.sps->size() << " bytes)\n";
        }
        if (vs.pps) {
            std::cout << "  PPS (decoded, " << vs.pps->size() << " bytes)\n";
        }
        if (vs.vps) {
            std::cout << "  VPS (decoded, " << vs.vps->size() << " bytes)\n";
        }

        for (size_t j = 0; j < vs.cryptoInfos.size(); ++j) {
            const auto &ci = vs.cryptoInfos[j];
            std::cout << "  Crypto #" << j + 1 << ":\n";
            std::cout << "    Tag: " << ci.tag << "\n";
            std::cout << "    Suite: " << ci.cryptoSuite << "\n";
            std::cout << "    Key Params (decoded, " << ci.keyParams.size() << " bytes)\n";
        }
    }
}
