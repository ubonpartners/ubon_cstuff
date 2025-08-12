#ifndef __MP4_WRITER_H
#define __MP4_WRITER_H

typedef struct mp4_writer mp4_writer_t;

// mp4 writer: very basic creation of fragmented MP4 from h264/h265 GOPs
// it's intended you call into this from the output of h26x assembler
// framerate/resolution automatically set from the stream data
// make sure you set length_prefixed=true on the h26x assembler create as
// mp4_writer expects length-prefixed NALU stream
mp4_writer_t *mp4_writer_create(const char *filename, bool h265);
// add one frame of NALUs
// format is length prefixed nalus i.e.
// some number of: (len>>24)&0xff, (len>>16)&0xff, (len>>8)&0xff, (len>>0)&0xff, <length 'len' nalu bytes>
void mp4_writer_add_video_frame(mp4_writer_t *wr, uint64_t extended_90khz_timestamp, uint8_t *data, int data_length);
void mp4_writer_destroy(mp4_writer_t *wr);

#endif