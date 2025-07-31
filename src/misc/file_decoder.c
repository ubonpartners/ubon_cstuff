#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include "pcap_stuff.h"
#include "rtp_receiver.h"
#include "h26x_assembler.h"
#include "simple_decoder.h"
#include "pcap_decoder.h"
#include "cuda_stuff.h"

bool file_decoder_file_ends(const char *file, const char *suffix)
{
    size_t file_len = strlen(file);
    size_t suffix_len = strlen(suffix);
    if (file_len < suffix_len) return false;
    return strcmp(file + file_len - suffix_len, suffix) == 0;
}

simple_decoder_codec_t file_decoder_parse_codec(const char *file)
{
    bool is_h264=file_decoder_file_ends(file, ".h264")||file_decoder_file_ends(file, ".264");
    if (is_h264) return SIMPLE_DECODER_CODEC_H264;
    bool is_h265=file_decoder_file_ends(file, ".h265")||file_decoder_file_ends(file, ".265")||file_decoder_file_ends(file, ".hevc");
    if (is_h265) return SIMPLE_DECODER_CODEC_H265;
    return SIMPLE_DECODER_CODEC_UNKNOWN;
}

float file_decoder_parse_fps(const char *s) {
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

typedef struct decode_file_state
{
    void *context;
    void (*callback)(void *context, image_t *img);
    simple_decoder_t *dec;
    h26x_assembler_t *a;
} decode_file_state_t;


static void decoder_callback(void *context, image_t *frame)
{
    decode_file_state_t *s=(decode_file_state_t *)context;
    s->callback(s->context, frame);
}

static void assembler_callback(void *context, const h26x_frame_descriptor_t *desc)
{
    decode_file_state_t *s=(decode_file_state_t *)context;
    simple_decoder_decode(s->dec, desc->annexb_data, desc->annexb_length, desc->extended_rtp_timestamp/90000.0);
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

    bool is_h264=file_decoder_file_ends(file, ".h264")||file_decoder_file_ends(file, ".264");
    bool is_h265=file_decoder_file_ends(file, ".h265")||file_decoder_file_ends(file, ".265")||file_decoder_file_ends(file, ".hevc");
    bool is_pcap=file_decoder_file_ends(file, ".pcap")||file_decoder_file_ends(file, ".pcapng");

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
            framerate=file_decoder_parse_fps(file);
            if (framerate==0)
            {
                log_warn("No framerate specified in filename %s, using default 30 fps", file);
                framerate=30.0f;
            }
            else
            {
                log_info("Found stream '%s' framerate %f fps", file, framerate);
            }
        }

        decode_file_state_t s;
        s.callback=callback;
        s.context=context;
        s.a=h26x_assembler_create(is_h265 ? H26X_CODEC_H265 : H26X_CODEC_H264, &s, assembler_callback);
        s.dec = simple_decoder_create(&s, decoder_callback, is_h265 ? SIMPLE_DECODER_CODEC_H265 : SIMPLE_DECODER_CODEC_H264);
        assert(s.dec!=0);
        FILE *f=fopen(file, "rb");
        if (!f)
        {
            log_error("Failed to open input file %s", file);
            return;
        }
        char buffer[4096];
        size_t bytes_read;
        while(1)
        {
            bool stop=false;
            while ((bytes_read = fread(buffer, 1, sizeof(buffer), f)) > 0)
            {
                int fr=h26x_assembler_process_raw_video(s.a, framerate, (uint8_t *)buffer, bytes_read);
                if (stop_callback)
                {
                    stop=stop_callback(context);
                    if (stop) break;
                }
            }
            if (stop) break;
            if (!stop_callback) break;
            fseek(f, 0, SEEK_SET); // loop
        }
        fclose(f);
        h26x_assembler_destroy(s.a);
        simple_decoder_destroy(s.dec);
        return;
    }

    log_error("Unsupported file format: %s", file);
}
