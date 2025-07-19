#include <iostream>
#include <vector>
#include <string>
#include <string.h>
#include <unistd.h>
#include "memory_stuff.h"
#include "cuda_stuff.h"
#include "misc.h"
#include "audioframe.h"
#include "audio_io.h"
#include "log.h"
#include "infer_aux.h"

int main(int argc, char *argv[]) {
    log_debug("ubon_cstuff version = %s", ubon_cstuff_get_version());
    init_cuda_stuff();

    wav_reader_t *wr=wav_reader_create("/mldata/video/wav/John_Wick_7dc33fe2bdb0_32000.wav");

    infer_aux_t *inf=infer_aux_create("/mldata/efficientat/trt/efficientat_m10_as.trt", 0);

    float d=wav_reader_get_wav_duration(wr);
    printf("WAV file length %f seconds\n",d);

    audioio_stream_params_t P;
    memset(&P, 0, sizeof(audioio_stream_params_t));
    P.mode = AUDIOIO_MODE_FULL_DUPLEX;
    P.capture_device = "default";
    P.playback_device = "default";
    P.sample_rate = 32000;
    P.channels = 1;
    P.frame_ms = 10;
    P.use_float = 1;
    audioio_stream_t *io = audioio_stream_create(&P);
    if (!io) { fprintf(stderr, "Failed: %s\n", audioio_strerror(io)); }

    // Pull recorded frames:
    printf("Starting audio\n");
    while (1) {
        usleep(10000);
        audioframe_t *in = audioio_capture_get_frame(io, 10);
        printf("Mic got frame %p LUFS %f\n",in,audioframe_compute_lufs_approx(in));

        if (in) { /* process */ audioframe_destroy(in); }
        // Provide playback frames (silence example)
        audioframe_t *out = wav_reader_get_audioframe(wr, 10);
        printf("WAV got frame %p LUFS %f\n",out,audioframe_compute_lufs_approx(out));
        if (!out) break;
        audioio_playback_write_frame(io, out);
        audioframe_destroy(out);
    }
    audioio_stream_destroy(io);

    /*while(1)
    {
        audioframe_t *fr=wav_reader_get_audioframe(wr, 10);
        printf("got frame\n");
        if (!fr) break;
        audioframe_destroy(fr);
    }*/
}