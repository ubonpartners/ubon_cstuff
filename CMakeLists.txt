cmake_minimum_required(VERSION 3.10)
project(uboncstuff LANGUAGES C CXX CUDA)
# When scikit-build-core drives CMake (inside `pip install`) it sets the
# variable `SKBUILD`; turn the apps OFF in that case so the wheel build
# only compiles the library + Python module.
option(BUILD_APPS "Build CLI test programs" ON)

if (DEFINED SKBUILD)
    message(STATUS "skbuild detected – turning BUILD_APPS OFF for wheel build")
    set(BUILD_APPS OFF)
endif()
# -------------------------------------------------------------------

message("")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(DEFINED ENV{JETSON_SOC})
    message(STATUS, "Enabling Orin Nano = UBONCSTUFF_PLATFORM=1")
else()
    message(STATUS, "Disabling Orin Nano = UBONCSTUFF_PLATFORM=0")
endif()

# Ensure Python3 is found first so Python3_EXECUTABLE is available
find_package(Python3 REQUIRED COMPONENTS Interpreter)


# Automatically detect pybind11 installed via pip or conda
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Run git describe and write to a header file
execute_process(
    COMMAND git describe --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE UBON_CSTUFF_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
message(STATUS "PYBIND11_CMAKE_DIR: ${PYBIND11_CMAKE_DIR}")

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 60 70 86 89)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)

# Use built-in CMake flags for different build types
# use cmake -DCMAKE_BUILD_TYPE=Debug
# or cmake -DCMAKE_BUILD_TYPE=Release

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -ggdb3 -fvar-tracking -fvar-tracking-assignments")
set(CMAKE_C_FLAGS_DEBUG  "-O0 -g3 -ggdb3 -fvar-tracking -fvar-tracking-assignments")
set(CMAKE_CUDA_FLAGS_DEBUG "-O1 -G -g")

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
set(CMAKE_C_FLAGS_RELEASE "-O2 -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -x c++")  # treat .c files as C++

# Path to the generated header
set(VERSION_HEADER ${CMAKE_BINARY_DIR}/ubon_cstuff_version.h)

# Write the version string to the header file
set(GUARD_NAME "_UBON_CSTUFF_VERSION_H_")
file(WRITE "${VERSION_HEADER}" "#ifndef ${GUARD_NAME}\n")
file(APPEND "${VERSION_HEADER}" "#define ${GUARD_NAME}\n")
file(APPEND "${VERSION_HEADER}" "#define UBON_CSTUFF_VERSION \"${UBON_CSTUFF_VERSION}\"\n")
file(APPEND "${VERSION_HEADER}" "#endif /* ${GUARD_NAME} */\n")

find_package(yaml-cpp REQUIRED)

# CUDA setup
enable_language(CUDA)
#find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89")

# Include dirs
set(PROJECT_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/track/bytetracker
    ${CMAKE_SOURCE_DIR}/src/track/utrack
    /usr/include/eigen3
    /usr/local/include/eigen3
    ${CUDA_INCLUDE_DIRS}
)

set(COMMON_LIBS
    -lpthread -lm -ldl -lrt
)

if(DEFINED ENV{JETSON_SOC})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUBONCSTUFF_PLATFORM=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUBONCSTUFF_PLATFORM=1")
    list(APPEND PROJECT_INCLUDE_DIRS
        /usr/src/jetson_multimedia_api/include
        /usr/src/jetson_multimedia_api/include/libjpeg-8b
        /usr/local/cuda/include
        /usr/include/libdrm
    )
    list(APPEND COMMON_LIBS
        jetson_multimedia_lib
        #/usr/local/cuda/lib64/stubs/libnvidia-ml.a
    )
    set(ARCH_LIBS
        -lSDL2 -ljpeg -lopenh264 -lyuv -lpcap -lsrtp2
        -lcuda -lcudart -lnvv4l2
        -lnppc -lnppisu -lnppig -lnppicc -lnppif -lnppist -lnppidei
        -lnvbufsurface -lnvbufsurftransform
        -lnvvpi -lnvjpeg -lnvdla_compiler
        -lnvinfer -lnvonnxparser -lnvinfer_plugin
        -Wl,-rpath=/usr/lib/aarch64-linux-gnu/nvidia
        -Wl,-rpath=/usr/local/cuda/lib64
        -Wl,--disable-new-dtags
        -L/usr/lib/aarch64-linux-gnu
        -L/usr/lib/aarch64-linux-gnu/nvidia
        -L/usr/lib/aarch64-linux-gnu/tegra
        -L/usr/local/cuda/lib64
        -L/usr/local/cuda/lib64/stubs
    )
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUBONCSTUFF_PLATFORM=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUBONCSTUFF_PLATFORM=0")
    list(APPEND PROJECT_INCLUDE_DIRS
        /usr/local/cuda/include
    )
    set(ARCH_LIBS
        -L/usr/local/cuda/lib64
        -lnvinfer -lcudart -lcublas -lnvonnxparser
        -lopenh264 -lcuda -lnvcuvid -lyuv -lpcap -lsrtp2
        -lnppc -lnppisu -lnppig -lnppicc -lnppif -lnppist -lnppidei -lSDL2 -ljpeg
        -lnvjpeg
        -lnvidia-ml
        -lnvidia-opticalflow
    )
endif()

include_directories(${PROJECT_INCLUDE_DIRS})

# Include the binary dir to access the generated header
include_directories(${CMAKE_BINARY_DIR} )

# Gather sources
file(GLOB SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/c_tests/*.c
    ${CMAKE_SOURCE_DIR}/src/cuda/*.c
    ${CMAKE_SOURCE_DIR}/src/decode/*.c
    ${CMAKE_SOURCE_DIR}/src/detections/*.c
    ${CMAKE_SOURCE_DIR}/src/image/*.c
    ${CMAKE_SOURCE_DIR}/src/infer/*.c
    ${CMAKE_SOURCE_DIR}/src/log/*.c
    ${CMAKE_SOURCE_DIR}/src/match/*.c
    ${CMAKE_SOURCE_DIR}/src/misc/*.c
    ${CMAKE_SOURCE_DIR}/src/motion_track/*.c
    ${CMAKE_SOURCE_DIR}/src/nvof/*.c
    ${CMAKE_SOURCE_DIR}/src/rtp/*.c
    ${CMAKE_SOURCE_DIR}/src/track/*.c
    ${CMAKE_SOURCE_DIR}/src/track/bytetracker/*.cpp
    ${CMAKE_SOURCE_DIR}/src/track/utrack/*.c
)

file(GLOB CUDA_FILES
    ${CMAKE_SOURCE_DIR}/src/cuda/*.cu
)

if(DEFINED ENV{JETSON_SOC})
file(GLOB JETSON_MULTIMEDIA_FILES
    /usr/src/jetson_multimedia_api/samples/common/classes/*.cpp
)
endif()

# Force -fPIC on all source files explicitly
foreach(source_file ${SRC_FILES})
    set_source_files_properties(${source_file} PROPERTIES COMPILE_FLAGS "-fPIC")
endforeach()

foreach(source_file ${CUDA_FILES})
    set_source_files_properties(${source_file} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

add_library(uboncstufflib STATIC ${SRC_FILES} ${CUDA_FILES})
set_target_properties(uboncstufflib PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DEFINED ENV{JETSON_SOC})
add_library(jetson_multimedia_lib STATIC ${JETSON_MULTIMEDIA_FILES})
set_target_properties(jetson_multimedia_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# --- Build apps -----------------------------------------
if (BUILD_APPS)
    set(APP_MAIN_FILES
        ${CMAKE_SOURCE_DIR}/src/main.c
        ${CMAKE_SOURCE_DIR}/src/apps/decode_benchmark.c
        ${CMAKE_SOURCE_DIR}/src/apps/decode_file.c
        ${CMAKE_SOURCE_DIR}/src/apps/infer_thread_test.c
        ${CMAKE_SOURCE_DIR}/src/apps/infer_benchmark.c
        ${CMAKE_SOURCE_DIR}/src/apps/pcap_play.c
        ${CMAKE_SOURCE_DIR}/src/apps/network_play.c
        ${CMAKE_SOURCE_DIR}/src/apps/run_tests.c
        ${CMAKE_SOURCE_DIR}/src/apps/track_benchmark.c
        ${CMAKE_SOURCE_DIR}/src/apps/track_test.c
        ${CMAKE_SOURCE_DIR}/src/apps/warp_test.c
    )

    foreach(app_src ${APP_MAIN_FILES})
        get_filename_component(app_name ${app_src} NAME_WE)
        add_executable(${app_name} ${app_src})
        target_link_libraries(${app_name}
            uboncstufflib
            yaml-cpp
            ${COMMON_LIBS}
            ${ARCH_LIBS}
        )
    endforeach()
endif()
# --------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

add_subdirectory(src/bindings)

install(TARGETS ubon_pycstuff
        LIBRARY DESTINATION python/ubon_pycstuff)  # install into package folder

# Clean target
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_BINARY_DIR}/uboncstuff")
