from setuptools import setup, Extension
from setuptools import find_packages
import pybind11
import numpy

ext_modules = [
    Extension(
        "ubon_pycstuff.ubon_pycstuff",
        ["src/ubon_pycstuff/ubon_pycstuff.cpp"],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            "../include"
        ],
        extra_objects=[
            "../build/libuboncstufflib.a",  
        ],
        libraries=[
            "pthread", 
            "cuda", 
            "cudart",
            "cublas",
            "nvinfer",
            "nvonnxparser",
            "openh264",
            "nvcuvid",
            "nppc",
            "nppisu",
            "nppig",
            "nppicc",
            "jpeg",
            "SDL2",
            "yuv",
            "nvidia-opticalflow"
        ],
        library_dirs=[
            "/usr/local/lib",       # common location for libyuv
            "/usr/lib",             # fallback
            "/usr/lib/x86_64-linux-gnu",  # on many Debian/Ubuntu systems
        ],
        language="c++"
    )
]

setup(
    name="ubon_pycstuff",
    version="0.1.0",
    author="Mark Blake",
    description="A Python module with C++ backend using pybind11",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    zip_safe=False,
)
