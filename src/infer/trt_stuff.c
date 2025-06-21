
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cassert>
#include <algorithm>
#include <cassert>
#include <memory>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <mutex>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include "NvInfer.h"
#include "trt_stuff.h"

static std::once_flag initFlag;
static allocation_tracker_t trt_alloc_tracker;
Logger trt_Logger;
TRTAllocator trt_allocator(&trt_alloc_tracker);

static void do_trt_init()
{
    log_debug("TRT init");
    allocation_tracker_register(&trt_alloc_tracker, "trt alloc", true);
}

void trt_init()
{
    std::call_once(initFlag, do_trt_init);
}