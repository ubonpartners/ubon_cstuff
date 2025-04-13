# 🛠️ Ubon_cstuff

**Useful C code**
- Surface wrapper abstraction supporting
  - host and GPU surfaces with various colour formata
  - Surface scaling (currently using NPP, libyuv) 
  - Format/colourspace converstion
- Video decoder supporting
  - OpenH264
  - NVDDEC
- Optical flow supporting
  - NVOF
  - VPI (WIP...)
- TensorRT inference

**Python bindings**
- All of above functionality exposed very simply
- Can see how to use it [here](https://github.com/ubonpartners/ubon_cstuff/blob/main/ubon_pycstuff/test.py)

**How to build**

- C part is built with cmake; usual mkdir build; cd build; cmake ..; make
- Needs various things libyuv, openh264, SDL, cuda,....
- to build python bindings: cd ubon_pycstuff; python setup.py build_ext --inplace
- add ubon_cstuff/ubon_pycstuff/src to your PYTHONPATH
  
## 📜 License

This project is licensed under the **[Ubon Cooperative License](https://github.com/ubonpartners/license/blob/main/LICENSE)**

Please reach out with questions:  
📬 [bernandocribbenza@gmail.com](mailto:bernandocribbenza@gmail.com?subject=yolo-dpa%20question&body=Your-code-is-rubbish!)
