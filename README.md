# 🛠️ Ubon_cstuff

**High-performance C modules with simple Python bindings**

This repo provides a set of modular, GPU-aware C components for image and video processing — with seamless Python integration via `pybind11`.

---

## 🔧 Features

### Surface Abstraction
- Unified wrapper for working with host and GPU image buffers
- Supports multiple color formats (RGB, YUV, planar, packed, etc.)
- GPU and CPU scaling using:
  - NVIDIA NPP
  - libyuv
- Color format & colorspace conversion

### Video Decoding
- Hardware and software decoder support:
  - OpenH264 (software)
  - NVDEC (NVIDIA GPU-accelerated decode)

### Optical Flow
- Support for:
  - NVIDIA Optical Flow SDK (NVOF)
  - VPI (in progress…)

### Inference
- TensorRT accelerated inference with support for batched processing and flexible surface input

---

## 🐍 Python Bindings

All of the above functionality is exposed via simple, direct Python bindings using `pybind11`.

See the usage example here:  
👉 [test.py](https://github.com/ubonpartners/ubon_cstuff/blob/main/test.py)

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

### Supported image conversions

🧭 Supported image conversions (✔️=direct, 🟧 =2-step)

```
| From \ To     | YUV420_DEV  | YUV420_HOST  | RGB24_DEV  | RGB24_HOST  | FP16_DEV  | FP16_HOST  | FP32_DEV | FP32_HOST |
|---------------|-------------|--------------|------------|-------------|-----------|------------|----------|-----------|
| NV12_DEV      | ✔️          |              |            |             |           |            |          |           |
| YUV420_DEV    |             | ✔️           |            | 🟧          | ✔️        |            | ✔️       |           |
| YUV420_HOST   | ✔️          |              |            | ✔️          |           |            |          |           |
| RGB24_DEV     |             |              |            | ✔️          |           |            |          |           |
| RGB24_HOST    | 🟧          | 🟧           | ✔️         |             |           |            |          |           |
| RGB_FP16_DEV  |             |              | ✔️         | 🟧          |           |            |          |           |
| RGB_FP16_HOST |             |              |            |             | ✔️        |            |          |           |
| RGB_FP32_DEV  |             |              |            |             |           |            |          | ✔️        |
| RGB_FP32_HOST |             |              |            |             |           |            | ✔️       |           |
```



## 📜 License

This project is licensed under the  
**[Ubon Cooperative License](https://github.com/ubonpartners/license/blob/main/LICENSE)**

---

## 💬 Questions?

Feel free to reach out:

📬 [bernandocribbenza@gmail.com](mailto:bernandocribbenza@gmail.com?subject=ubon_cstuff%20question&body=Your-code-is-rubbish!)

