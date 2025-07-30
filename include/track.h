#ifndef __TRACK_H
#define __TRACK_H

#include <vector>

typedef struct track_shared_state track_shared_state_t;
typedef struct track_stream track_stream_t;
typedef struct track_results track_results_t;;

#include "image.h"
#include "detections.h"
#include "simple_decoder.h"
#include "infer.h"

typedef enum result_type
{
    TRACK_FRAME_SKIP_FRAMERATE=0,           // time gap since previous processed frame too small
    TRACK_FRAME_SKIP_NO_MOTION=1,           // motion analysis determined not enough change
    TRACK_FRAME_SKIP_NO_IMG=2,              // 0 was passed in for 'img'
    TRACK_FRAME_TRACKED_ROI=3,              // normally processed frame
    TRACK_FRAME_TRACKED_FULL_REFRESH=4,     // periodic full ROI
} result_type_t;

struct track_results
{
    result_type_t result_type;
    double time;
    roi_t motion_roi;                // ROI that was detected as containing motion. Note that if this is
                                     // less than skip_thr then inference/tracking will be skipped. In this
                                     // case inference_roi will be (0,0,0,0) and track_dets/inference_dets 0
    roi_t inference_roi;             // ROI that was used for inference (likely larger than motion_roi)
    detection_list_t *track_dets;        // tracker output
    detection_list_t *inference_dets;    // raw detection output (for debug)
};

//=================================================================================================
// shared state is things that live across multiple streams. For example inference model
// setup. One config is supplied which sets a lot of stuff

track_shared_state_t *track_shared_state_create(const char *yaml_config);
void track_shared_state_destroy(track_shared_state_t *tss);

// one 'track_stream' should be created per camera, video clip, audio clip or whatever separate thing you are processing
track_stream_t *track_stream_create(track_shared_state_t *tss, void *result_context, void (*result_callback)(void *context, track_results_t *results),
                                    const char *config_yaml=0);
void track_stream_destroy(track_stream_t *ts);

const char *track_shared_state_get_stats(track_shared_state_t *tss);
const char *track_stream_get_stats(track_stream_t *ts);

//=================================================================================================
// packet interface- set an 'sdp' -actually looks at the first thing that looks like
// an m-line and also SRTP config- can be H264,H265 or OPUS

void track_stream_set_sdp(track_stream_t *ts, const char *sdp_str);
// receive an RTP packet into the track_stream - all decryption, reordering etc, handled
void track_stream_add_rtp_packets(track_stream_t *ts, int num_packets, uint8_t **data, int *length);

//=================================================================================================
// jpeg interface. You can create an extra 'stream' and just use it for jpegs
// for example, you might want to process a user uploaded image in order to get the face rec
// embedding to use for search. You are guaranteed to get one result_callback per JPEG
// note: jpeg decode of arbitrary images is dangerous - suggest transcoding unsafe input
// in a separate cloud process before passing into this function
bool track_stream_run_on_jpeg(track_stream_t *ts, uint8_t *jpeg_data, int jpeg_data_length);

//=================================================================================================
// stream config - these things are already defaulted in the shared state yaml config
// some extra functions are provided for cases you want to override on a per-stream basis
// min_process sets minimum time (in seconds) between successive frames where inference is run
// i.e. specifies the maximum framerate to run the inference at
// min_full_ROI is an interval that specifies how often to ignore the motiontracker and run on the whole frame
// - this is useful to stop things getting "stuck".
void track_stream_set_minimum_frame_intervals(track_stream_t *ts, double min_process, double min_full_ROI);
// enabled face-rec embedding generation
void track_stream_enable_face_embeddings(track_stream_t *ts, bool enabled, float min_quality);

//=================================================================================================
// advanced / lower level interfaces
// can mostly ignore this unless you want to write some more specialist kind of app or test app
model_description_t *track_shared_state_get_model_description(track_shared_state_t *tss);
void track_shared_state_configure_inference(track_shared_state_t *tss, infer_config_t *config);
// returns the preferred image format for this stream
image_format_t track_stream_get_stream_image_format(track_stream_t *ts);
// most low-level run interface, can specify img and seperate time (img can be null to just advance time)
void track_stream_run(track_stream_t *ts, image_t *img, double time);
// simple run interface which processes a whole .264 or .265 stream
void track_stream_run_video_file(track_stream_t *ts, const char *file, simple_decoder_codec_t codec, double video_fps, bool loop_forever=false);
// run interface which takes time from img->meta.timestamp - be sure it's monotonic!
void track_stream_run_frame_time(track_stream_t *ts, image_t *img);
// run inference on a single independent frame
void track_stream_run_single_frame(track_stream_t *ts, image_t *img);
// if you do not provide a result callback then the results are accumulated into a vector
// (one entry per frame) and can be retrieved with the below
std::vector<track_results_t *> track_stream_get_results(track_stream_t *ts);
track_results_t *track_results_create();
void track_results_destroy(track_results_t *tr);


#endif
