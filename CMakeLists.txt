cmake_minimum_required(VERSION 3.10)
project(uboncstuff LANGUAGES C CXX CUDA)

# Ensure Python3 is found first so Python3_EXECUTABLE is available
find_package(Python3 REQUIRED COMPONENTS Interpreter)


# Automatically detect pybind11 installed via pip or conda
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
message(STATUS "PYBIND11_CMAKE_DIR: ${PYBIND11_CMAKE_DIR}")

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 60 70 86 89)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)

set(CMAKE_CXX_FLAGS "-O2 -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -x c++")  # treat .c files as C++

find_package(yaml-cpp REQUIRED)

# CUDA setup
enable_language(CUDA)
#find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89")

# Include dirs
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Gather sources
file(GLOB SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/c_tests/*.c
    ${CMAKE_SOURCE_DIR}/src/cuda/*.c
    ${CMAKE_SOURCE_DIR}/src/decode/*.c
    ${CMAKE_SOURCE_DIR}/src/image/*.c
    ${CMAKE_SOURCE_DIR}/src/infer/*.c
    ${CMAKE_SOURCE_DIR}/src/log/*.c
    ${CMAKE_SOURCE_DIR}/src/misc/*.c
    ${CMAKE_SOURCE_DIR}/src/nvof/*.c
    ${CMAKE_SOURCE_DIR}/src/vpi/*.c
)

file(GLOB CUDA_FILES
    ${CMAKE_SOURCE_DIR}/src/cuda/*.cu
)

# Force -fPIC on all source files explicitly
foreach(source_file ${SRC_FILES})
    set_source_files_properties(${source_file} PROPERTIES COMPILE_FLAGS "-fPIC")
endforeach()

foreach(source_file ${CUDA_FILES})
    set_source_files_properties(${source_file} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

set(MAIN_FILE ${CMAKE_SOURCE_DIR}/src/main.c)

add_library(uboncstufflib STATIC ${SRC_FILES} ${CUDA_FILES})
set_target_properties(uboncstufflib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Build the main executable from main + static lib
add_executable(uboncstuff ${MAIN_FILE})

target_link_libraries(uboncstuff
    uboncstufflib
    yaml-cpp
    -L/usr/local/cuda/lib64
    -lnvinfer -lcudart -lcublas -lpthread -lnvonnxparser
    -lopenh264 -lcuda -lnvcuvid -lyuv
    -lnppc -lnppisu -lnppig -lnppicc -lnppif -lSDL2 -ljpeg
    -lnvvpi
    -lnvidia-opticalflow
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

add_subdirectory(src/bindings)

install(TARGETS ubon_pycstuff
        LIBRARY DESTINATION python/ubon_pycstuff)  # install into package folder

# Clean target
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_BINARY_DIR}/uboncstuff")
