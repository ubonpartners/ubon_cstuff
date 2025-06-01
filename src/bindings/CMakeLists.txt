add_library(ubon_pycstuff MODULE ubon_pycstuff.cpp)
target_include_directories(ubon_pycstuff PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(DEFINED ENV{JETSON_SOC})
    set(ARCH_LIBS
        pthread SDL2 jpeg openh264 yuv pcap srtp2
        cuda cudart nvv4l2
        nppc nppisu nppig nppicc nppif nppidei
        nvbufsurface nvbufsurftransform
        nvvpi nvinfer nvonnxparser nvinfer_plugin
        -L/usr/lib/aarch64-linux-gnu
        -L/usr/lib/aarch64-linux-gnu/nvidia
        -L/usr/lib/aarch64-linux-gnu/tegra
        -L/usr/local/cuda/lib64
    )
else()
    set(ARCH_LIBS
        pthread cuda cudart cublas nvinfer nvonnxparser openh264 pcap srtp2
        nvcuvid nppc nppisu nppig nppicc nppif nppidei jpeg SDL2 yuv
        nvidia-opticalflow yaml-cpp # etc
    )
endif()

target_link_libraries(ubon_pycstuff PRIVATE
    uboncstufflib  # your static lib
    pybind11::module
    Python3::Python
    ${ARCH_LIBS}
)

# Determine if we're doing a normal local build or a wheel build
#if(EXISTS "${CMAKE_SOURCE_DIR}/python/ubon_pycstuff")
    # Local development — put .so directly in Python package
#    set(BINDINGS_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/python/ubon_pycstuff")
#else()
    # Wheel build — use normal binary dir
#    set(BINDINGS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
#endif()
set(BINDINGS_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/ubon_pycstuff")

set_target_properties(ubon_pycstuff PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BINDINGS_OUTPUT_DIR}"
    PREFIX ""  # Don't prepend 'lib'
    SUFFIX ".so"  # Ensure proper module extension
)
