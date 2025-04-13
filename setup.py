from skbuild import setup
from setuptools import find_packages

setup(
    name="ubon_pycstuff",
    version="0.1.0",
    description="Python bindings for ubon C code using pybind11",
    author="Mark Blake",
    packages=find_packages("python"),
    package_dir={"": "python"},
    include_package_data=True,
)

