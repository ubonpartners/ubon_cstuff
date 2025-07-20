# 🛠️ Ubon_cstuff

**High-performance C modules with simple Python bindings**

This repo provides a set of modular, GPU-aware C components for image and video processing — with seamless Python integration via `pybind11`.

---

## 🔧 Features

### Surface Abstraction
- Unified wrapper for working with host and GPU image buffers
- Reference counted immutable design makes it easy to pass surface handles between threads
- Supports multiple color formats (RGB, YUV, planar, packed, mono, etc.)
- it's generally intended most 'work' is done in YUV420 -and this will usually be most efficient- although operations should generally work on any format, with automatic conversion
- GPU and CPU scaling using:
  - NVIDIA NPP
  - libyuv
  - factorized scaling for large downscale factors (repeated /2, then NPP for last scale)
- Color format & colorspace conversion
- **Fully asynchronous use of cuda with automatic chaining and synchronization**
- Additional operations like crop, blend, hash, gaussian blur, mean absolute difference
- write surface to JPEG, or display on screen

### Network/framing
- RTP packet receiver supporting re-ordering, ssrc/pt validation filtering
- SRTP decryption support
- Re-assembly of video packets into complete H264/H265 frames
- PCAP file parsing support for test apps

### Video Decoding
- Platforms Supported
  - Desktop x86_64 with Nvidia card => UBONCSTUFF_PLATFORM=0
  - Integrated Nvidia Orin Nano  => UBONCSTUFF_PLATFORM=1
- Hardware and software decoder support:
  - OpenH264 (software)
  - NVDEC (NVIDIA GPU-accelerated decode) H264/H265

### Optical Flow
- Support for:
  - NVIDIA Optical Flow SDK (NVOF); latest v3 support with NV12 or mono
  - VPI (in progress…) based on https://docs.nvidia.com/vpi/installation.html

### Inference
- TensorRT accelerated inference with support for batched processing and flexible surface input

---

## Prerequisities
- sudo apt install nvidia-cuda-toolkit
- TensorRT installed (libnvinfer10, via apt)
- sudo apt install libyuv-dev
- sudo apt install libsdl2-dev
- sudo apt install libyaml-cpp-dev
- sudo apt install pybind11-dev
- sudo apt install libopenh264-dev
- sudo apt install libsrtp2-dev
- sudo apt install libpcap-dev
- sudo apt install libeigen3-dev
- sudo apt install libjpeg-dev
- sudo apt install libnvjpeg-dev-12-6
- sudo apt install libnvvpi3 vpi3-dev vpi3-samples
- sudo apt install libfftw3-dev
- sudo apt install libasound-dev

---

## 🐍 Python Bindings

All of the above functionality is exposed via simple, direct Python bindings using `pybind11`.
- Dependencies:
  - pip install -r requirements.txt
  - Install the latest [Stuff][git@github.com:ubonpartners/stuff.git]
  - Install this package ubon_cstuff either using pip or wheel

See the usage example here:  
👉 [The examples](https://github.com/ubonpartners/ubon_cstuff/blob/main/examples.py)
👉 [The tests](https://github.com/ubonpartners/ubon_cstuff/blob/main/test.py)

## Prebuilt python wheel

[Download here (note might be out of date)](https://drive.google.com/file/d/1VTfBOdzqFs8tkpScRHW-Df6PeWfW0r5U/view?usp=sharing)

After installing you should be able to run the examples/tests above

---

## 🔨 Build Instructions

### 🧱 C++ Core

```
mkdir build
cd build
cmake ..
make
```

> 🔧 You’ll need development versions of:  
> `libyuv`, `openh264`, `SDL2`, `CUDA`, `TensorRT`, etc.

### 🐍 Python Bindings

From the root of the repo (`ubon_cstuff/`):

- To install in editable dev mode:
  ```
  pip install -e .
  ```

- To build a Python wheel:
  ```
  python -m build
  ```

---

## Other stuff

### Supported image formats

🧭 Supported image formats
- Direct conversion with optimized kernel or NPP is supported for many cases
- Almost all cases at least work
- Some format conversions can be done with out any real work through reference counting and pointer tricks (e.g. YUV->MONO)

```
IMAGE_FORMAT_YUV420_HOST
IMAGE_FORMAT_YUV420_DEVICE
IMAGE_FORMAT_NV12_DEVICE
IMAGE_FORMAT_RGB24_HOST
IMAGE_FORMAT_RGB24_DEVICE
IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE
IMAGE_FORMAT_RGB_PLANAR_FP16_HOST
IMAGE_FORMAT_RGB_PLANAR_FP32_HOST
IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE
IMAGE_FORMAT_MONO_HOST
IMAGE_FORMAT_MONO_DEVICE
```



## 📜 License

This project is licensed under the  
**[Ubon Cooperative License](https://github.com/ubonpartners/license/blob/main/LICENSE)**

---

## 💬 Questions?

Feel free to reach out:

📬 [bernandocribbenza@gmail.com](mailto:bernandocribbenza@gmail.com?subject=ubon_cstuff%20question&body=Your-code-is-rubbish!)

