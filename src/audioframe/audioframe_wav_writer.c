// ------------------------------------------------------------
// wav_writer.c
// ------------------------------------------------------------
// Minimal WAV writer for audioframe_t sequences.
// Assumptions / Design:
//   * All audioframes passed to wav_writer_write_audioframe() share the same
//     sample rate & channel count. The first frame establishes file format.
//   * Frames contain normalized float samples in [-1,1]. We clamp & quantize
//     to 16-bit signed PCM little-endian for disk.
//   * Writer streams: we reserve space for headers, write data incrementally,
//     then patch RIFF/WAVE sizes in wav_writer_destroy().
//   * Simplicity > features: only PCM 16-bit output implemented; extendable.
//   * Thread-safety: not thread-safe. Serialize external calls if needed.
//
// Public interface (already declared in audioframe.h):
//   typedef struct wav_writer wav_writer_t;
//   wav_writer_t *wav_writer_create(const char *filename);
//   void wav_writer_destroy(wav_writer_t *wr);
//   void wav_writer_write_audioframe(wav_writer_t *wr, audioframe_t *fr);
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Forward declaration of audioframe API (already included if compiling together)
#include "audioframe.h"

#ifndef WAV_WRITER_BITS_PER_SAMPLE
#define WAV_WRITER_BITS_PER_SAMPLE 16
#endif

struct wav_writer {
    FILE *f;
    int sample_rate;
    int num_channels;
    int bytes_per_sample;     // derived from bits_per_sample
    int header_written;       // 1 after first frame defines format & header emitted
    int64_t data_bytes_written; // payload size (not including header)
};

static int write_u16_le(FILE *f, uint16_t v) {
    unsigned char b[2];
    b[0] = (unsigned char)(v & 0xFF);
    b[1] = (unsigned char)((v >> 8) & 0xFF);
    return (fwrite(b,1,2,f)==2)?0:-1;
}
static int write_u32_le(FILE *f, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xFF);
    b[1] = (unsigned char)((v >> 8) & 0xFF);
    b[2] = (unsigned char)((v >> 16) & 0xFF);
    b[3] = (unsigned char)((v >> 24) & 0xFF);
    return (fwrite(b,1,4,f)==4)?0:-1;
}

static void wav_writer_emit_header(wav_writer_t *wr) {
    // Emit a placeholder header; we'll come back to patch sizes.
    // RIFF chunk descriptor
    fwrite("RIFF",1,4,wr->f);
    write_u32_le(wr->f, 0); // placeholder for chunk size (4 + (8+fmt) + (8+data))
    fwrite("WAVE",1,4,wr->f);

    // fmt subchunk (PCM)
    fwrite("fmt ",1,4,wr->f);
    write_u32_le(wr->f, 16);              // Subchunk1Size for PCM
    write_u16_le(wr->f, 1);               // AudioFormat = PCM
    write_u16_le(wr->f, (uint16_t)wr->num_channels);
    write_u32_le(wr->f, (uint32_t)wr->sample_rate);
    uint32_t byte_rate = (uint32_t)wr->sample_rate * wr->num_channels * wr->bytes_per_sample;
    write_u32_le(wr->f, byte_rate);
    uint16_t block_align = (uint16_t)(wr->num_channels * wr->bytes_per_sample);
    write_u16_le(wr->f, block_align);
    write_u16_le(wr->f, WAV_WRITER_BITS_PER_SAMPLE);

    // data subchunk header placeholder
    fwrite("data",1,4,wr->f);
    write_u32_le(wr->f, 0); // placeholder for data chunk size

    wr->header_written = 1;
}

wav_writer_t *wav_writer_create(const char *filename) {
    if (!filename) return NULL;
    FILE *f = fopen(filename, "wb");
    if (!f) return NULL;
    wav_writer_t *wr = (wav_writer_t*)calloc(1,sizeof(*wr));
    if (!wr) { fclose(f); return NULL; }
    wr->f = f;
    wr->bytes_per_sample = WAV_WRITER_BITS_PER_SAMPLE / 8;
    wr->data_bytes_written = 0;
    return wr;
}

static void wav_writer_patch_sizes(wav_writer_t *wr) {
    if (!wr || !wr->header_written) return;
    long file_end = ftell(wr->f);
    // RIFF chunk size = file_size - 8
    long riff_size = file_end - 8;
    // data chunk size = data_bytes_written
    long data_chunk_size = wr->data_bytes_written;

    // Patch RIFF size at offset 4
    fseek(wr->f, 4, SEEK_SET);
    write_u32_le(wr->f, (uint32_t)riff_size);

    // Patch data chunk size: after 'data' tag at: 12 (RIFF) + (8+16) fmt = 36 => offset 40
    // Layout: 0..3 'RIFF', 4..7 size, 8..11 'WAVE', 12..15 'fmt ', 16..19 fmt size (16),
    // 20..35 fmt body (16 bytes), 36..39 'data', 40..43 data size
    fseek(wr->f, 40, SEEK_SET);
    write_u32_le(wr->f, (uint32_t)data_chunk_size);

    fseek(wr->f, file_end, SEEK_SET); // return to end (not strictly needed now)
}

void wav_writer_destroy(wav_writer_t *wr) {
    if (!wr) return;
    if (wr->f) {
        if (wr->header_written) {
            wav_writer_patch_sizes(wr);
        }
        fclose(wr->f);
    }
    free(wr);
}

void wav_writer_write_audioframe(wav_writer_t *wr, audioframe_t *fr) {
    if (!wr || !fr) return;
    int fr_rate = audioframe_get_sample_rate(fr);
    int fr_ch   = audioframe_get_num_channels(fr);
    int fr_frames = audioframe_get_num_samples(fr);
    float *data = audioframe_get_data(fr);

    if (!wr->header_written) {
        // Initialize format from first frame
        wr->sample_rate = fr_rate;
        wr->num_channels = fr_ch;
        wav_writer_emit_header(wr);
    } else {
        // Validate consistency
        assert(fr_rate == wr->sample_rate);
        assert(fr_ch == wr->num_channels);
    }

    // Convert float [-1,1] to 16-bit PCM
    size_t total_samples = (size_t)fr_frames * wr->num_channels;
    // To avoid large stack allocations for huge frames, process in chunks
    const size_t CHUNK = 4096;
    int16_t *buf = (int16_t*)malloc(CHUNK * sizeof(int16_t));
    if (!buf) return; // allocation failure: silently drop (or handle differently)

    size_t processed = 0;
    while (processed < total_samples) {
        size_t n = total_samples - processed;
        if (n > CHUNK) n = CHUNK;
        for (size_t i=0;i<n;i++) {
            float v = data[processed + i];
            if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
            int sample = (int)lrintf(v * 32767.0f);
            if (sample < -32768) sample = -32768; // (shouldn't happen after clamp)
            buf[i] = (int16_t)sample;
        }
        fwrite(buf, sizeof(int16_t), n, wr->f);
        processed += n;
    }
    free(buf);

    wr->data_bytes_written += (int64_t)total_samples * wr->bytes_per_sample;
}

// ------------------------------------------------------------
// End wav_writer.c
// ------------------------------------------------------------
