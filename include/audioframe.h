#ifndef __AUDIOFRAME_H
#define __AUDIOFRAME_H

typedef struct audioframe audioframe_t;

audioframe_t *audioframe_create(int num_samples, int sample_rate=16000, int num_channels=1);
audioframe_t *audioframe_reference(audioframe_t *fr);
void audioframe_destroy(audioframe_t *fr);
float *audioframe_get_data(audioframe_t *fr);
int audioframe_get_sample_rate(audioframe_t *fr);
int audioframe_get_num_channels(audioframe_t *fr);
int audioframe_get_num_samples(audioframe_t *fr);

float audioframe_compute_peak(audioframe_t *fr);
float audioframe_compute_rms(audioframe_t *fr);
float audioframe_compute_energy(audioframe_t *fr);
float audioframe_compute_lufs_approx(audioframe_t *fr);

typedef struct wav_reader wav_reader_t;
wav_reader_t *wav_reader_create(const char *filename);
void wav_reader_destroy(wav_reader_t *wr);
float wav_reader_get_wav_duration(wav_reader_t *wr); // returns duration in seconds
audioframe_t *wav_reader_get_audioframe(wav_reader_t *wr, int num_ms=10); // returns "num_ms" of next audio, might not be exactly this, returns 0 when no audio left

typedef struct audioframe_resampler audioframe_resampler_t;
audioframe_resampler_t *audioframe_resampler_create(int target_sample_rate, int target_num_channels);
void audioframe_resampler_destroy(audioframe_resampler_t *r);
audioframe_t *audioframe_resample(audioframe_resampler_t *r, audioframe_t *fr, int num_ms); // returns at least num_ms of audio, 0 returned if not enough audio yet

typedef struct wav_writer wav_writer_t;
wav_writer_t *wav_writer_create(const char *filename);
void wav_writer_destroy(wav_writer_t *wr);
void wav_writer_write_audioframe(wav_writer_t *wr, audioframe_t *fr);

#endif