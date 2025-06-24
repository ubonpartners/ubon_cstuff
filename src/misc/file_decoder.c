#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include "pcap_stuff.h"
#include "rtp_receiver.h"
#include "h26x_assembler.h"
#include "simple_decoder.h"
#include "pcap_decoder.h"
#include "cuda_stuff.h"

static bool file_ends(const char *file, const char *suffix)
{
    size_t file_len = strlen(file);
    size_t suffix_len = strlen(suffix);
    if (file_len < suffix_len) return false;
    return strcmp(file + file_len - suffix_len, suffix) == 0;
}

static float parse_fps(const char *s) {
    for (const char *p = s; *p; ++p) {
        if (*p == '_' && isdigit((unsigned char)p[1])) {
            char *endptr;
            float val = strtof(p + 1, &endptr);
            if (endptr != p + 1 && strncmp(endptr, "fps", 3) == 0) {
                return val;
            }
        }
    }
    return 0.0f;
}

// decode a file, including .h264, .h265, .pcap, .pcapng
// if framerate is 0, it will try to parse the framerate from the filename
// if framerate is still 0, it will use a default of 30 fps
// the callback will be called for each decoded frame
// context is passed to the callback
// the callback should not destroy the image, it will be destroyed by the decoder
void decode_file(const char *file, void *context,
                 void (*callback)(void *context, image_t *img),
                 float framerate,
                 bool (*stop_callback)(void *context))
{
    rtp_receiver_t *rtp_receiver;
    h26x_assembler_t *h26x_assembler;
    simple_decoder_t *decoder;

    bool is_h264=file_ends(file, ".h264")||file_ends(file, ".264");
    bool is_h265=file_ends(file, ".h265")||file_ends(file, ".265");
    bool is_pcap=file_ends(file, ".pcap")||file_ends(file, ".pcapng");

    if (is_pcap)
    {
        pcap_decoder_t *pcap_dec=pcap_decoder_create(file);
        while(1)
        {
            image_t *img=pcap_decoder_get_frame(pcap_dec);
            if (!img) break;
            callback(context, img);
            if (stop_callback)
            {
                bool stop=stop_callback(context);
                if (stop) break;
            }
        }
        pcap_decoder_destroy(pcap_dec);
        return;
    }
    if (is_h264 || is_h265)
    {
        if (framerate==0)
        {
            framerate=parse_fps(file);
            if (framerate==0)
            {
                log_warn("No framerate specified in filename %s, using default 30 fps", file);
                framerate=30.0f;
            }
        }

        simple_decoder_t *dec = simple_decoder_create(context, callback, is_h265 ? SIMPLE_DECODER_CODEC_H265 : SIMPLE_DECODER_CODEC_H264);
        assert(dec!=0);
        simple_decoder_set_framerate(dec, framerate);
        FILE *f=fopen(file, "rb");
        if (!f)
        {
            log_error("Failed to open input file %s", file);
            return;
        }
        char buffer[4096];
        size_t bytes_read;
        while ((bytes_read = fread(buffer, 1, sizeof(buffer), f)) > 0)
        {
            simple_decoder_decode(dec, (uint8_t *)buffer, bytes_read);
            if (stop_callback)
            {
                bool stop=stop_callback(context);
                if (stop) break;
            }
        }
        fclose(f);
        simple_decoder_destroy(dec);
        return;
    }

    log_error("Unsupported file format: %s", file);
}
