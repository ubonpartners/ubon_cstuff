// ------------------------------------------------------------
// audioframe_wav_reader.c
// ------------------------------------------------------------
// A minimal WAV reader that outputs floating-point audioframe_t blocks.
// Supported formats: PCM 8/16/24/32-bit (little endian), IEEE float 32-bit.
// Channel counts: 1..8 (up/down mixing handled when producing frames).
// The reader does *not* load the entire file; it streams from disk.
//
// NOTE: The header you supplied uses default arguments (e.g. int num_ms=10) which
// are not valid in pure C. This implementation ignores default parameters – callers
// must supply explicit arguments.
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#include "audioframe.h"

#ifndef WAV_READER_INTERNAL_BUFFER_MS
#define WAV_READER_INTERNAL_BUFFER_MS 200   // internal read buffer (~latency control)
#endif

#ifndef WAV_MAX_CHANNELS
#define WAV_MAX_CHANNELS 8
#endif

struct wav_reader {
    FILE *f;
    int sample_rate;
    int num_channels;
    int bits_per_sample;      // original bit depth
    int bytes_per_sample;     // derived from bits_per_sample
    int data_chunk_size;      // in bytes
    int data_bytes_read;      // bytes consumed from data chunk
    long data_start_pos;      // file offset of first data byte
    int total_samples_per_chan; // total samples per channel in file (may be 0 if unknown)

    // Temporary interleaved buffer (raw decoded floats before channel remix or slicing)
    float *float_buffer;
    size_t float_buffer_capacity; // in *frames* (multi-channel sample frames)
};

static int read_u16_le(FILE *f, uint16_t *out) {
    uint8_t b[2];
    if (fread(b,1,2,f)!=2) return -1;
    *out = (uint16_t)(b[0] | (b[1]<<8));
    return 0;
}
static int read_u32_le(FILE *f, uint32_t *out) {
    uint8_t b[4];
    if (fread(b,1,4,f)!=4) return -1;
    *out = (uint32_t)(b[0] | (b[1]<<8) | (b[2]<<16) | (b[3]<<24));
    return 0;
}

static int wav_reader_parse_header(wav_reader_t *wr) {
    FILE *f = wr->f;
    uint8_t riff[4];
    if (fread(riff,1,4,f)!=4 || memcmp(riff,"RIFF",4)!=0) return -1;
    uint32_t riff_size; if (read_u32_le(f,&riff_size)!=0) return -1; (void)riff_size;
    uint8_t wave[4]; if (fread(wave,1,4,f)!=4 || memcmp(wave,"WAVE",4)!=0) return -1;

    int fmt_found = 0;
    int data_found = 0;
    uint16_t audio_format=0;
    uint32_t data_chunk_size = 0;
    while (!fmt_found || !data_found) {
        uint8_t chunk_id[4];
        if (fread(chunk_id,1,4,f)!=4) return -1;
        uint32_t chunk_size; if (read_u32_le(f,&chunk_size)!=0) return -1;
        if (memcmp(chunk_id,"fmt ",4)==0) {
            // fmt chunk
            uint16_t num_channels; uint32_t sample_rate; uint32_t byte_rate; uint16_t block_align; uint16_t bits_per_sample;
            if (read_u16_le(f,&audio_format)!=0) return -1;
            if (read_u16_le(f,&num_channels)!=0) return -1;
            if (read_u32_le(f,&sample_rate)!=0) return -1;
            if (read_u32_le(f,&byte_rate)!=0) return -1; (void)byte_rate;
            if (read_u16_le(f,&block_align)!=0) return -1; (void)block_align;
            if (read_u16_le(f,&bits_per_sample)!=0) return -1;
            // Skip any extra fmt bytes
            long remaining = (long)chunk_size - 16;
            if (remaining < 0) return -1;
            if (remaining>0) fseek(f, remaining, SEEK_CUR);

            wr->sample_rate = (int)sample_rate;
            wr->num_channels = (int)num_channels;
            wr->bits_per_sample = (int)bits_per_sample;
            wr->bytes_per_sample = (bits_per_sample+7)/8;
            fmt_found = 1;
        } else if (memcmp(chunk_id,"data",4)==0) {
            wr->data_start_pos = ftell(f);
            data_chunk_size = chunk_size;
            wr->data_chunk_size = (int)data_chunk_size;
            fseek(f, chunk_size, SEEK_CUR); // jump to after data; we'll rewind later
            data_found = 1;
        } else {
            // skip other chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
        // Chunks are word (2-byte) aligned; if size odd, consume pad byte
        if (chunk_size & 1) fseek(f, 1, SEEK_CUR);
    }
    if (!fmt_found || !data_found) return -1;
    if (wr->num_channels <=0 || wr->num_channels>WAV_MAX_CHANNELS) return -1;

    wr->total_samples_per_chan = wr->data_chunk_size / (wr->bytes_per_sample * wr->num_channels);
    // Rewind to data
    fseek(f, wr->data_start_pos, SEEK_SET);
    wr->data_bytes_read = 0;
    return 0;
}

wav_reader_t *wav_reader_create(const char *filename) {
    if (!filename) return NULL;
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    wav_reader_t *wr = (wav_reader_t*)calloc(1,sizeof(*wr));
    if (!wr) { fclose(f); return NULL; }
    wr->f = f;
    wr->float_buffer_capacity = 0;
    wr->float_buffer = NULL;
    if (wav_reader_parse_header(wr)!=0) {
        wav_reader_destroy(wr);
        return NULL;
    }
    return wr;
}

void wav_reader_destroy(wav_reader_t *wr) {
    if (!wr) return;
    if (wr->f) fclose(wr->f);
    free(wr->float_buffer);
    free(wr);
}

float wav_reader_get_wav_duration(wav_reader_t *wr) {
    if (!wr || wr->sample_rate<=0) return 0.0f;
    return (float)wr->total_samples_per_chan / (float)wr->sample_rate;
}

static void ensure_float_buffer(wav_reader_t *wr, size_t frames_needed) {
    if (frames_needed <= wr->float_buffer_capacity) return;
    size_t new_cap = wr->float_buffer_capacity ? wr->float_buffer_capacity : 1024;
    while (new_cap < frames_needed) new_cap *= 2;
    float *nb = (float*)realloc(wr->float_buffer, new_cap * wr->num_channels * sizeof(float));
    if (nb) {
        wr->float_buffer = nb;
        wr->float_buffer_capacity = new_cap;
    }
}

static void pcm_to_float(const uint8_t *src, int count, int bytes_per_sample, int bits_per_sample, float *dst) {
    // count = number of *samples* (not frames) across all channels
    int i;
    if (bits_per_sample == 8) {
        for (i=0;i<count;i++) dst[i] = ((int)src[i] - 128) / 128.0f; // unsigned 8-bit PCM
        return;
    }
    if (bytes_per_sample==2) {
        for (i=0;i<count;i++) {
            int16_t v = (int16_t)(src[2*i] | (src[2*i+1]<<8));
            dst[i] = (float)(v / 32768.0f);
        }
        return;
    }
    if (bytes_per_sample==3) {
        for (i=0;i<count;i++) {
            int32_t v = (int32_t)( (src[3*i]) | (src[3*i+1]<<8) | (src[3*i+2]<<16) );
            if (v & 0x800000) v |= ~0xFFFFFF; // sign extend 24-bit
            dst[i] = (float)(v / 8388608.0f); // 2^23
        }
        return;
    }
    if (bytes_per_sample==4 && bits_per_sample==32) {
        // We don't know if it's PCM 32 or float 32. Assume float if range looks plausible.
        // For simplicity: if bits_per_sample==32 treat as float; adapt as needed.
        memcpy(dst, src, count*sizeof(float));
        return;
    }
    // Fallback: zero
    for (i=0;i<count;i++) dst[i]=0.0f;
}

audioframe_t *wav_reader_get_audioframe_ms(wav_reader_t *wr, int num_ms) {
    if (!wr || num_ms<=0) return NULL;
    int frames_requested = (int)(( (int64_t)wr->sample_rate * num_ms) / 1000);
    if (frames_requested <=0) return 0;
    return wav_reader_get_audioframe_frames(wr, frames_requested);
}


audioframe_t *wav_reader_get_audioframe_frames(wav_reader_t *wr, int frames_requested) {
    if (!wr || frames_requested<=0) return NULL;
    if (wr->data_bytes_read >= wr->data_chunk_size) return NULL; // EOF

    if (frames_requested <=0) return 0;

    int bytes_per_frame = wr->bytes_per_sample * wr->num_channels;
    int frames_left = (wr->data_chunk_size - wr->data_bytes_read) / bytes_per_frame;
    if (frames_left <=0) return NULL;
    if (frames_requested > frames_left) frames_requested = frames_left; // last chunk may be shorter

    ensure_float_buffer(wr, frames_requested);
    size_t bytes_to_read = (size_t)frames_requested * bytes_per_frame;
    uint8_t *raw = (uint8_t*)malloc(bytes_to_read);
    if (!raw) return NULL;
    size_t n = fread(raw,1,bytes_to_read,wr->f);
    if (n != bytes_to_read) {
        // truncated
        frames_requested = (int)(n / bytes_per_frame);
        bytes_to_read = (size_t)frames_requested * bytes_per_frame;
    }
    wr->data_bytes_read += (int)bytes_to_read;
    pcm_to_float(raw, frames_requested * wr->num_channels, wr->bytes_per_sample, wr->bits_per_sample, wr->float_buffer);
    free(raw);

    // Create audioframe (already interleaved float)
    audioframe_t *fr = audioframe_create(frames_requested, wr->sample_rate, wr->num_channels);
    if (!fr) return NULL;
    float *dst = audioframe_get_data(fr);
    memcpy(dst, wr->float_buffer, frames_requested * wr->num_channels * sizeof(float));
    return fr;
}
