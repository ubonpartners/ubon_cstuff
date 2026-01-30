# Project Overview

**ubon_cstuff** is a high-performance C/C++/CUDA library for image and video processing, with Python bindings via **pybind11**. It targets realtime and offline scenarios on **x86 + NVIDIA GPU** and **Jetson Orin Nano**, providing:

- **Surface and image utilities** – unified host/GPU buffers, colour conversion, scaling (NPP, libyuv), and async CUDA chaining.
- **Networking** – RTP reception, SRTP decryption, H26x frame assembly, PCAP parsing.
- **Video decoding** – hardware (NVDEC, Jetson) and software (OpenH264) for H.264/H.265 and JPEG.
- **Optical flow** – NVIDIA Optical Flow (NVOF) on x86; VPI on Jetson.
- **Inference** – TensorRT object detection and auxiliary models (face, CLIP, FIQA, audio) with batching and mixed precision.
- **Tracking** – motion-based ROI, ByteTrack and utrack, optional embeddings/JPEGs via `track_aux`.

Many components run asynchronously with thread pools and work queues; the tracking pipeline ties decode, motion, inference, and tracking together with automatic backpressure and rate limiting.

---

## Codebase Map

Directory layout and responsibility of each area. Public API lives in `include/`; implementation is under `src/` with the same logical grouping.

```
ubon_cstuff/
├── CMakeLists.txt          # Build: library, apps, Python bindings
├── setup.py / pyproject.toml
├── include/                # Public C/C++ headers (API surface)
├── src/
│   ├── bindings/           # pybind11 Python module (ubon_pycstuff)
│   ├── apps/               # CLI test/benchmark executables (optional, BUILD_APPS)
│   ├── audio_alg/          # Audio event preprocessing (EfficientAT, mel filter, etc.)
│   ├── audioframe/         # Audio frame I/O (ALSA, WAV, resampler)
│   ├── config/             # Config loading
│   ├── c_tests/            # C-level test hooks
│   ├── cuda/               # CUDA kernels (convert, scale, NMS, Jetson helpers)
│   ├── decode/             # Video decoders (NVDEC, Jetson, OpenH264) + NAL headers
│   ├── detections/         # Detection lists, matching, embeddings, FIQA
│   ├── image/              # Image create/convert/scale/draw/display, JPEG, webcam
│   ├── infer/              # TensorRT inference + infer_thread (async)
│   ├── log/                # Logging
│   ├── match/              # IoU matching, MOTA metrics
│   ├── misc/               # Hash, YAML, work_queue, Kalman, ROI, solvers, platform
│   ├── motion_track/       # Motion tracker (optical flow → ROI)
│   ├── nvof/               # NVIDIA Optical Flow (CUDA + Jetson backends)
│   ├── rtp/                # RTP receiver, H26x assembler, SDP, PCAP, MP4 writer
│   ├── track/              # Tracking pipeline (see Architecture)
│   │   ├── bytetracker/     # ByteTrack (C++)
│   │   ├── utrack/         # Lightweight tracker
│   │   ├── track_stream.c  # Stream entry (RTP/NAL/JPEG/file)
│   │   ├── track_shared.c  # Shared state, workers, config
│   │   ├── track_aux.c     # Embeddings, JPEGs, aux models
│   │   └── trackset.c       # Ground-truth YAML load/replay
│   └── vpi/                # VPI optical flow (Jetson)
├── ubon_pycstuff/          # Python package (__init__.py; native module built from bindings)
├── docs/
├── examples.py
└── test.py
```

### Include headers (by domain)

| Domain        | Key headers |
|---------------|-------------|
| Image/surface | `image.h`, `image_draw.h`, `jpeg.h`, `display.h`, `webcam.h` |
| Decode       | `simple_decoder.h` |
| RTP/network  | `rtp_receiver.h`, `h26x_assembler.h`, `sdp_parser.h`, `pcap_decoder.h`, `mp4_writer.h` |
| Inference    | `infer.h`, `infer_thread.h`, `infer_aux.h` |
| Detections   | `detections.h`, `embedding.h`, `fiqa.h` |
| Optical flow | `nvof.h`, `vpi.h` |
| Motion       | `motion_track.h` |
| Tracking     | `track.h`, `trackset.h` |
| Audio        | `audio_event.h`, `audioframe.h`, `audio_io.h` |
| Misc         | `config.h`, `log.h`, `memory_stuff.h`, `work_queue.h`, `yaml_stuff.h`, `roi.h`, `kalman_tracker.h` |
| Match/eval   | `match.h`, `mota_metrics.h` |

### Apps (when `BUILD_APPS=ON`)

Built from `src/apps/` and `src/main.c`; used for testing and benchmarking:

- `audio_test`, `decode_benchmark`, `decode_file`, `infer_benchmark`, `infer_thread_test`
- `mp4_test`, `nal_play`, `network_play`, `of_test`, `pcap_play`, `rt_benchmark`
- `run_tests`, `stress_test`, `track_benchmark`, `track_test`, `warp_test`

---

## Platforms

- **UBONCSTUFF_PLATFORM=0** – x86_64 with NVIDIA GPU (CUDA, NVDEC, NVOF, TensorRT, NPP, libyuv, OpenH264, etc.).
- **UBONCSTUFF_PLATFORM=1** – Jetson Orin Nano (`JETSON_SOC` set): Jetson multimedia API, VPI optical flow, same TensorRT/inference stack.

Decoders and optical flow backends are chosen per platform; the rest of the API is shared.

---

## Python bindings (ubon_pycstuff)

The full C/CUDA API is exposed in Python via **pybind11**: same performance as calling the C library, with NumPy interop and no C++ required. The native module is built from `src/bindings/` and installed as the `ubon_pycstuff` package.

**Install:** From the repo root, `pip install -r requirements.txt` then `pip install -e .` (see root [README](../README.md) for system prerequisites). Alternatively use a [prebuilt wheel](https://drive.google.com/file/d/1VTfBOdzqFs8tkpScRHW-Df6PeWfW0r5U/view?usp=sharing) if available for your platform.

**Import and minimal example:**

```python
import ubon_pycstuff.ubon_pycstuff as upyc
import numpy as np
from PIL import Image

# Load image, copy to GPU, scale, copy back to NumPy
rgb = np.asarray(Image.open("image.jpg").convert("RGB"))
img = upyc.c_image.from_numpy(rgb)
img = img.scale(1280, 720)
result = img.to_numpy()
```

**Exposed API (summary):**

| Area | Classes / functions |
|------|---------------------|
| **Images** | `c_image`, `from_numpy`, `scale`, `convert`, `to_numpy`, `crop`, `blur`, `mad_4x4`, `display`, `load_jpeg`, `decode_jpeg` |
| **Inference** | `c_infer` (sync `run` / `run_batch`), `c_infer_thread` (async `infer_async` / `wait`), `c_infer_aux` (face/CLIP/FIQA) |
| **Tracking** | `c_track_shared`, `c_track_stream` — `run_on_images`, `run_on_video_file`, `run_on_jpeg`, `set_sdp`, `add_rtp_packets`, `get_results` |
| **Decode** | `c_decoder` — H.264/H.265 bitstream → list of `c_image` |
| **Optical flow** | `c_nvof` — run on `c_image` → cost and flow NumPy arrays |
| **Motion** | `c_motion_tracker` — `add_frame`, `get_roi`, `set_roi`, `get_of_results` |
| **Evaluation** | `c_pcap_decoder`, `c_mota_metrics`, `match_box_iou` |
| **Utilities** | `c_get_aligned_faces`, `KalmanBoxTracker`, `c_platform_stats`, `get_version`, `log_set_level`, `cuda_set_sync_mode`, file trace |

Runnable scripts: **examples.py** and **test.py** in the repo root (inference, decode, optical flow, motion, tracking, etc.).

---

## Build and install

- **C/C++/CUDA library and optional apps**: `mkdir build && cd build && cmake .. && make`. Set `BUILD_APPS=OFF` (or use a wheel build) to skip CLI apps.
- **Python**: from repo root, `pip install -e .` (editable) or `python -m build` for a wheel.

Prerequisites (see root README): CUDA, TensorRT, libyuv, SDL2, yaml-cpp, pybind11, OpenH264, libsrtp2, libpcap, Eigen, libjpeg, libnvjpeg; on Jetson add VPI, libasound, libfftw3, libsamplerate.

---

## Summary

ubon_cstuff is a modular, GPU-aware toolkit for video and image pipelines: surfaces, decode, RTP, inference, tracking, and optional audio/embeddings. The **track** subsystem is the main integration point for live or file-based video: one shared state (YAML config, workers, inference) and one stream per source (camera, file, RTP, JPEG), with results delivered via callbacks or the Python `get_results` API.
