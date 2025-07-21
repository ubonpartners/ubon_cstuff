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
    bool in_media = false;
    enum { NONE, VIDEO, AUDIO } media_type = NONE;

    while (std::getline(sdp, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Detect media line: m=video or m=audio
        if (line.rfind("m=video", 0) == 0 || line.rfind("m=audio", 0) == 0) {
            // Push previous
            if (in_media) {
                if (media_type == VIDEO) parsed->videoStreams.push_back(current);
                else if (media_type == AUDIO) parsed->audioStreams.push_back(current);
            }

            // Start new
            current = RtpParameters();
            // default clock for video, override on rtpmap
            current.clockRate = 90000;
            in_media = true;
            media_type = (line.rfind("m=video", 0) == 0) ? VIDEO : AUDIO;

            std::istringstream iss(line.substr(2));  // skip "m="
            std::string media, proto;
            int port, pt;
            iss >> media >> port >> proto >> pt;
            current.port = port;
            current.payloadType = pt;
            continue;
        }

        if (!in_media) continue;

        // a=rtpmap: payload codec/clock[/channels]
        if (line.rfind("a=rtpmap:", 0) == 0) {
            std::regex rtpmap(R"(^a=rtpmap:(\d+)\s+([^/]+)/([0-9]+)(?:/(?:[0-9]+))?)");
            std::smatch m;
            if (std::regex_search(line, m, rtpmap)) {
                int pt = std::atoi(m[1].str().c_str());
                if (pt == current.payloadType) {
                    current.codec = m[2];
                    current.clockRate = std::stoul(m[3]);
                }
            }
            continue;
        }

        // a=crypto for SRTP
        if (line.rfind("a=crypto:", 0) == 0) {
            std::regex crypto(R"(^a=crypto:(\d+)\s+(\S+)\s+inline:([A-Za-z0-9+/=]+))");
            std::smatch m;
            if (std::regex_search(line, m, crypto)) {
                RtpCryptoInfo ci;
                ci.tag = m[1];
                ci.cryptoSuite = m[2];
                ci.keyParams = base64_decode(m[3]);
                current.cryptoInfos.push_back(ci);
            }
            continue;
        }

        // H264 fmtp param sets (only for video)
        if (media_type == VIDEO && line.rfind("a=fmtp:", 0) == 0) {
            std::regex spsRx(R"(sprop-parameter-sets=([^,]+),([^;]+))");
            std::regex vpsRx(R"(sprop-vps=([^;]+))");
            std::smatch m;
            if (std::regex_search(line, m, spsRx)) {
                current.sps = base64_decode(m[1]);
                current.pps = base64_decode(m[2]);
            }
            if (std::regex_search(line, m, vpsRx)) {
                current.vps = base64_decode(m[1]);
            }
            continue;
        }
    }

    // Push last
    if (in_media) {
        if (media_type == VIDEO) parsed->videoStreams.push_back(current);
        else if (media_type == AUDIO) parsed->audioStreams.push_back(current);
    }

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
        std::cout << "Video Stream #" << i+1 << ":\n";
        std::cout << "  Port: " << vs.port << "\n";
        std::cout << "  Payload Type: " << vs.payloadType << "\n";
        std::cout << "  Codec: " << vs.codec << " @ " << vs.clockRate << " Hz\n";
        // ... SPS/PPS/VPS & Crypto omitted for brevity
    }
    for (size_t i = 0; i < parsed->audioStreams.size(); ++i) {
        const auto &as = parsed->audioStreams[i];
        std::cout << "Audio Stream #" << i+1 << ":\n";
        std::cout << "  Port: " << as.port << "\n";
        std::cout << "  Payload Type: " << as.payloadType << "\n";
        std::cout << "  Codec: " << as.codec << " @ " << as.clockRate << " Hz\n";
        // Crypto if present
    }
}
