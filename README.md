# ubon_cstuff

**GPU-accelerated video and image pipelines — in C/CUDA, with Python.**

A single library that takes live RTP streams, video files, or JPEGs and gives you **object detection, tracking, embeddings, and optional audio events** — with minimal glue code. Built for realtime (e.g. edge cameras) and offline; runs on **x86 + NVIDIA GPU** and **Jetson Orin Nano**.

---

## Why use it?

- **One pipeline, many inputs** — RTP (with SRTP), H.264/H.265 files, JPEG uploads, or raw frames. Same tracking and inference stack everywhere.
- **Realtime without the pain** — Thread pools, work queues, and automatic backpressure. Motion-based ROI and frame skipping keep latency and GPU load under control.
- **Detection + tracking + embeddings** — TensorRT object detection, ByteTrack or lightweight utrack, optional face/CLIP/FIQA and JPEG snapshots per track. Audio event detection (e.g. EfficientAT) when you need it.
- **First-class Python bindings** — The entire C/CUDA API is exposed in Python via **pybind11**: same performance, NumPy-friendly (`from_numpy` / `to_numpy`), no C++ required. Use it for scripting, pipelines, or prototyping; [examples](examples.py) and [tests](test.py) are in the repo; a [prebuilt wheel](https://drive.google.com/file/d/1VTfBOdzqFs8tkpScRHW-Df6PeWfW0r5U/view?usp=sharing) is available. Full API summary: [docs → Python bindings](docs/overview.md#python-bindings-ubon_pycstuff).

Details (codebase map, architecture) live in the **[docs](docs/)** folder: [Overview](docs/overview.md) · [Architecture](docs/architecture.md).

---

## Quick start

**Prerequisites** (Debian/Ubuntu): CUDA toolkit, TensorRT, libyuv, SDL2, yaml-cpp, pybind11, OpenH264, libsrtp2, libpcap, Eigen, libjpeg, libnvjpeg, **GCC 11** (build uses `gcc-11`/`g++-11`); on Jetson add VPI; for audio add libasound, libfftw3, libsamplerate. Example:

```bash
sudo apt install gcc-11 g++-11 nvidia-cuda-toolkit libyuv-dev libsdl2-dev \
  libyaml-cpp-dev pybind11-dev libopenh264-dev libsrtp2-dev libpcap-dev \
  libeigen3-dev libjpeg-dev libnvjpeg-dev-12-6 libfftw3-dev libasound-dev libsamplerate-dev
# TensorRT: libnvinfer10 (or libnvinfer-dev)
# libnvjpeg: package is versioned to CUDA (e.g. -12-6 for CUDA 12); pick the one matching your install
# Jetson only: libnvvpi3 vpi3-dev vpi3-samples
```

**Build (C/C++ core):**

```bash
mkdir build && cd build && cmake .. && make
```

Optional: `cmake -DBUILD_APPS=OFF ..` builds only the library and Python module (e.g. for wheel builds).

**Python (editable install):**

```bash
pip install -r requirements.txt
pip install -e .
```

Then run the [examples](examples.py) or [tests](test.py). **Python users**: you can use a [prebuilt wheel](https://drive.google.com/file/d/1VTfBOdzqFs8tkpScRHW-Df6PeWfW0r5U/view?usp=sharing) (may be outdated) to skip building the C++ core; see [Python bindings](docs/overview.md#python-bindings-ubon_pycstuff) in the docs for the API.

Build options (e.g. `BUILD_APPS=OFF` for library-only): [docs/overview.md](docs/overview.md#build-and-install).

---

## License

This project is dual-licensed under:

- **AGPL v3.0** — see [LICENSE](LICENSE) in this repo
- **[Ubon Cooperative License](https://github.com/ubonpartners/license/blob/main/LICENSE)**

You may use, modify, and distribute the software under either license.

Questions: [bernandocribbenza@gmail.com](mailto:bernandocribbenza@gmail.com?subject=ubon_cstuff%20question)
