# Architecture

This document describes the internal design of ubon_cstuff: the surface abstraction, decoding, RTP/network path, inference, optical flow and motion, tracking pipeline, audio path, and how the Python bindings expose them.

---

## 1. Surface and image abstraction

**Location**: `include/image.h`, `src/image/*.c`, `src/cuda/*.cu`

Images are represented as **reference-counted** `image_t` surfaces that can live on host or device. Formats include YUV420, NV12, RGB24, planar FP16/FP32, mono (Y-only), and tensor layouts. All buffers are immutable from the caller’s perspective; operations return new surfaces.

- **Creation / conversion**: `image_create`, `image_convert` (format and/or host↔device). Conversions use NPP or custom CUDA kernels where beneficial; some are zero-copy (e.g. YUV→mono via plane views).
- **Scaling**: `image_scale` – NPP or libyuv; for large downscales, factorized scaling (repeated ÷2 then NPP) is used.
- **Ops**: crop, blend, hash, Gaussian blur, 4×4 mean absolute difference (MAD). JPEG encode/decode and display (SDL2) operate on these surfaces.
- **Async CUDA**: Each surface can carry a `cudaStream_t`. Work is chained so dependencies are respected and the API can be used asynchronously; `image_sync()` blocks until all pending work on that surface is done.

This abstraction is the common type for all downstream stages: decode output, inference input, optical flow input, and tracking/aux inputs.

---

## 2. Video decoding

**Location**: `include/simple_decoder.h`, `src/decode/*.c`

**simple_decoder** turns H.264/H.265 bitstream (and optionally JPEG) into a sequence of `image_t` frames delivered via callback. One decoder instance per stream.

- **Platforms**:  
  - **x86**: NVDEC (NVCUVID) or OpenH264 (software).  
  - **Jetson**: Jetson multimedia API (hardware decode).
- **Input**: Length-prefixed NAL units (AVCC/HVCC). The RTP path reassembles RTP payloads into such NALUs before feeding the decoder.
- **Output format**: Configurable (e.g. YUV420 device); can be constrained (max width/height, minimum time delta) for rate limiting.
- **Modes**: Low-latency vs default; `force_skip` and time-based throttling reduce load when used behind motion or track_stream rate limiting.

Decoded frames are the primary input to the tracking pipeline and to standalone decode benchmarks/tools.

---

## 3. RTP, assembly, and network path

**Location**: `include/rtp_receiver.h`, `include/h26x_assembler.h`, `include/sdp_parser.h`, `src/rtp/*.c`

- **rtp_receiver**: Receives RTP packets, reorders by sequence number, validates SSRC/PT, optionally decrypts SRTP, and outputs in-order packets via callback. Stats (missing, duplicate, decrypt fail, etc.) are available.
- **h26x_assembler**: Consumes in-order RTP payloads (or NALUs) and reassembles full H.264/H.265 NAL units. Handles aggregation and fragmentation; outputs are suitable for `simple_decoder` (length-prefixed).
- **sdp_parser**: Parses SDP to get codec (H.264/H.265/OPUS), payload type, and SRTP keys so the track_stream can configure decoder and SRTP.
- **pcap_decoder / pcap_stuff**: Read RTP from PCAP files for testing; **mp4_writer** can write encoded or raw output to MP4.

In the tracking pipeline, **track_stream** uses these pieces: set SDP, then feed RTP packets; internally it drives the assembler → decoder → motion/inference/tracking.

---

## 4. Inference

**Location**: `include/infer.h`, `include/infer_thread.h`, `include/infer_aux.h`, `src/infer/*.c`

- **infer**: TensorRT engine load, model description (class names, keypoints, attributes, input/output dims, FP16/INT8). Single image or **batch** inference; automatic batching and precision handling. Output is a **detection_list_t** (boxes, classes, confidences, optional keypoints, REID, etc.).
- **infer_thread**: Async wrapper around the same engine: submit images (and optional ROI) via `infer_async`, wait by handle for detections and timings. Used by the tracking pipeline so decode and inference can overlap.
- **infer_aux**: Auxiliary TensorRT models (face embedding, CLIP, FIQA, etc.). Inputs are crops or preprocessed tensors; outputs are embeddings or per-attribute scores. Used by **track_aux** to attach embeddings/quality to tracked objects.

Inference is configured from YAML (thresholds, NMS, use_cuda_nms, fuse_face_person, batch/size limits). The main detector runs at the heart of the tracking pipeline; aux models run after tracking in track_aux.

---

## 5. Detections, embeddings, FIQA

**Location**: `include/detections.h`, `include/embedding.h`, `include/fiqa.h`, `src/detections/*.c`

- **detection_t**: Normalized box, class, confidence, track_id, optional subbox (e.g. face in person), keypoints (face/pose), attributes, REID vector, FIQA score, and optional pointers to **embedding_t** and **jpeg_t** (filled by track_aux).
- **detection_list_t**: Model description plus array of detections and timestamp.
- **embedding**: Opaque vector (e.g. face, CLIP) for similarity; **fiqa** provides face quality scores and optional FIQA embeddings.

Matching (IoU, Hungarian) and MOTA metrics live in **match**; detection lists are the interface between inference, trackers, and aux post-processing.

---

## 6. Optical flow and motion tracking

**Location**: `include/nvof.h`, `include/vpi.h`, `include/motion_track.h`, `src/nvof/*`, `src/vpi/*`, `src/motion_track/*`

- **NVOF (x86)**: NVIDIA Optical Flow SDK – takes a pair of frames (NV12 or mono), returns flow vectors and cost. Used for motion-based ROI and for motion score in the tracking pipeline.
- **VPI (Jetson)**: Optical flow on Jetson; motion_track can use either NVOF or VPI depending on platform.
- **motion_track**: Maintains state across frames: feed frames via `add_frame`; it runs optical flow and produces a **motion ROI** (and optional flow result array). The tracking pipeline uses this ROI to decide where to run inference and to skip frames when motion is below threshold.

This gives “only run detector where there’s motion” and “skip low-motion frames” behaviour without documenting low-level flow formats in the main track API.

---

## 7. Tracking subsystem

**Location**: `include/track.h`, `src/track/*.c`, `src/track/bytetracker/*.cpp`, `src/track/utrack/*.c`

The tracking pipeline is the main integration point for video: one **track_shared_state** (workers, CUDA, inference engines, config) and one **track_stream** per source (camera, file, RTP, JPEG).

### 7.1 Data flow

```text
Input (RTP / H26x NALUs / JPEG / video file / raw frames)
          ↓
    track_stream
      ├─ Decode (if needed) & rate limit
      ├─ Motion tracker (optical flow → ROI, skip if low motion)
      ├─ Main inference (detector on ROI or full frame)
      └─ Tracker (ByteTrack or utrack)
              ↓
        track_aux (optional: face/CLIP/FIQA, JPEG thumbnails)
              ↓
        result_callback(track_results_t)
```

Frames move through work queues so decode, inference, and aux can overlap; **track_shared** holds the thread pool and CUDA context so multiple streams share resources.

### 7.2 Components

- **track_stream**: Entry for all video data. Supports:
  - RTP: `track_stream_set_sdp` + `track_stream_add_rtp_packets`
  - NALUs: `track_stream_add_nalus` (length-prefixed)
  - JPEG: `track_stream_run_on_jpeg`
  - Video file: `track_stream_run_on_video_file`
  - Raw images: `track_stream_run_on_images` / `run_on_individual_images`
  Config: frame intervals, name; optional cancel_all_work. Results come via callback or, in Python, `get_results(wait)`.
- **track_shared**: Creates and owns shared state from a YAML config: worker threads, inference setup, decoder defaults. Multiple streams can be created from one shared state.
- **track_aux**: Post-processing for tracks: run auxiliary models (face, CLIP, FIQA), generate JPEG snapshots, attach embeddings to detections. Optional; configured via the same YAML/track_stream config.
- **trackset**: Load ground-truth tracks from YAML for evaluation or replay; separate from the live pipeline.
- **Algorithms**:
  - **bytetracker**: ByteTrack (Kalman + association); C++ implementation in `src/track/bytetracker/`.
  - **utrack**: Lighter-weight tracker (position prediction + motion integration).

### 7.3 track_results_t

Each callback receives **track_results_t**: result_type (skip framerate/motion/no_img, or tracked ROI/full refresh), time, motion_score, motion_roi, inference_roi, track_dets (tracker output), inference_dets (raw detector output for debug).

---

## 8. Audio path

**Location**: `include/audio_event.h`, `include/audioframe.h`, `src/audio_alg/*`, `src/audioframe/*`

- **audioframe**: Capture (ALSA), file I/O (WAV), resampling. Frames are passed to the audio-event model.
- **audio_alg**: Preprocessing for EfficientAT (and similar): mel filter, CUDA preprocess, etc. Produces input tensors for the event model.
- **audio_event**: Wraps an audio event model (EfficientAT or TinyCLAP); `audio_event_process` takes an audioframe and returns an embedding; postprocess returns top-k class indices/probs. Used for “what sound is this?” style events.

Audio is not wired into the main track_stream; it’s a separate pipeline that can be driven from Python or C with the same embedding/detection patterns as vision.

---

## 9. Python bindings

**Location**: `src/bindings/ubon_pycstuff.cpp`, `ubon_pycstuff/`

pybind11 builds a single native module that mirrors the main C API:

- **c_image**: Wraps `image_t`; constructors, `from_numpy`, scale/convert/crop/blend/blur/mad_4x4, to_numpy, display, sync, hash. JPEG: `load_jpeg`, `decode_jpeg`.
- **c_infer / c_infer_thread / c_infer_aux**: Construct from TRT + YAML; run, run_batch, configure, get_model_description; async path with infer_async/wait/get_stats.
- **c_track_shared / c_track_stream**: Shared state from YAML; stream supports set_frame_intervals, set_name, run_on_images, run_on_video_file, run_on_jpeg, set_sdp, add_rtp_packets, get_results, get_stats, cancel_all_work.
- **c_decoder**: Codec enum + decode(bitstream) → list of c_image.
- **c_nvof**: width/height + run(c_image) → cost and flow arrays.
- **c_motion_tracker**: Config YAML; add_frame, get_roi, set_roi, get_of_results.
- **c_pcap_decoder**: Open PCAP file; get_frame() → next decoded frame or None.
- **c_mota_metrics**, **match_box_iou**, **KalmanBoxTracker**: Evaluation and box matching.
- **Utilities**: c_get_aligned_faces, c_platform_stats, get_version, log_set_level, cuda_set_sync_mode, file_trace.

Numpy arrays are used for from_numpy, to_numpy, and for flow/cost/ROI lists; detection and track results are exposed as dicts/lists compatible with Python.

---

## 10. Build and library layout

- **CMake**: One static library **uboncstufflib** from all `src/*.c`, `src/track/*.c`, `src/track/bytetracker/*.cpp`, `src/track/utrack/*.c`, and `src/cuda/*.cu`. Optional **BUILD_APPS** adds executables in `src/apps/` plus `main.c`. Bindings are a separate target that links uboncstufflib and produces the Python extension.
- **Platform**: `UBONCSTUFF_PLATFORM=0` (x86) vs `1` (Jetson) selects decode, NVOF vs VPI, and library paths (e.g. Jetson multimedia, VPI, nvjpeg).
- **Dependencies**: CUDA, TensorRT, yaml-cpp, libyuv, OpenH264, libsrtp2, libpcap, SDL2, libjpeg, Eigen; on Jetson, VPI and ALSA/samplerate/fftw for audio.

This architecture keeps a single, consistent surface type from decode through inference and tracking, with clear separation between network/decode, inference, motion, tracking, and aux post-processing, and exposes the same capabilities from C and Python.
