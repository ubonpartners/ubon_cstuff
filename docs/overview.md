# Project Overview

`ubon_cstuff` provides a collection of high‑performance C/C++/CUDA modules for image and video processing.  Python bindings are
exposed through `pybind11`, allowing the low‑level components to be used directly from Python applications.

The codebase includes modules for:

- **Surface and Image Utilities** – unified handling of host and GPU buffers with automatic colour conversion and scaling.
- **Networking** – RTP packet reception, SRTP decryption and H26x frame assembly.
- **Video Decoding** – hardware and software decoders for H264/H265 and JPEG, supporting NVIDIA GPUs and Jetson devices.
- **Inference** – TensorRT‑accelerated object detection and auxiliary models (face, CLIP, FIQA, audio) with automatic batch and
  precision handling.
- **Tracking** – object tracking pipelines combining motion tracking, detectors and algorithms such as ByteTrack.

The project targets realtime scenarios but can scale down for offline processing.  Many components are written to run
asynchronously and make full use of available GPU resources.
