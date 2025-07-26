#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include "simple_decoder.h"
#include <string.h>
#include "memory_stuff.h"
#include "cuda_stuff.h"
#include "misc.h"

// Helper: combine frame hashes into a running total
static uint64_t combine_hash(uint64_t total, uint32_t frame_hash) {
    // simple mix: multiply by prime and add
    return total * 1315423911u + frame_hash;
}

struct ThreadResult {
    double fps;
    double mbps;
    bool compute_hash;
    uint64_t decoded_frames;
    uint64_t decoded_macroblocks;
    uint64_t total_hash;
};

static void process_image(void *context, image_t *img) {
    ThreadResult *out=(ThreadResult *)context;
    if (out->compute_hash)
    {
        uint32_t hash=image_hash(img);
        out->total_hash = combine_hash(out->total_hash, hash);
    }
    out->decoded_frames+=1;
    out->decoded_macroblocks += (img->width * img->height) / (16 * 16);
}

static void do_decode(const char *filename, bool is_h265, ThreadResult *out, int num_iters, bool compute_hash) {
    // Open input file
    cuda_thread_init();
    FILE *input = fopen(filename, "rb");
    if (!input) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }
    out->compute_hash = compute_hash;

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<num_iters;i++)
    {
        decode_file(filename, out, process_image, 0);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(end - start).count();
    out->fps = out->decoded_frames / secs;
    out->mbps = (double)out->decoded_macroblocks / secs;

    fclose(input);
}

typedef struct test
{
    const char *name;
    const char *video_file;
    int num_threads;
    int num_iters;
    bool compute_hash;
} test_t;

static test_t tests[]={
    {"UK office 1512p H264", "/mldata/video/test/uk_off_2688x1512_12.5fps.264", 1, 3, true},
    {"UK office 1512p H265", "/mldata/video/test/uk_off_2688x1512_12.5fps.265", 1, 3, true},
    {"INof 720p, H264", "/mldata/video/test/ind_off_1280x720_7.5fps.264", 1,   3, true},
    {"INof 720p, H265", "/mldata/video/test/ind_off_1280x720_7.5fps.265", 1,   3, true},
    {"BC1, 1080p, H264", "/mldata/video/test/bc1_1920x1080_30fps.264", 1, 1, true},
    {"BC1, 1080p, H265", "/mldata/video/test/bc1_1920x1080_30fps.265", 1, 1, true},

    {"UK office 1512p H264", "/mldata/video/test/uk_off_2688x1512_12.5fps.264", 8, 2, false},
    {"UK office 1512p H265", "/mldata/video/test/uk_off_2688x1512_12.5fps.265", 8, 2, false},
    {"INof 720p, H264", "/mldata/video/test/ind_off_1280x720_7.5fps.264", 8,   5, false},
    {"INof 720p, H265", "/mldata/video/test/ind_off_1280x720_7.5fps.265", 8,   5, false},
    {"BC1, 1080p, H264", "/mldata/video/test/bc1_1920x1080_30fps.264", 1, 1, false},
    {"BC1, 1080p, H265", "/mldata/video/test/bc1_1920x1080_30fps.265", 1, 1, false},
    {"BC1, 1080p, H264", "/mldata/video/test/bc1_1920x1080_30fps.264", 2, 1, false},
    {"BC1, 1080p, H265", "/mldata/video/test/bc1_1920x1080_30fps.265", 2, 1, false},
};

int main(int argc, char *argv[]) {
    log_debug("ubon_cstuff version = %s", ubon_cstuff_get_version());
    init_cuda_stuff();
    log_debug("Init cuda done");
    log_debug("Initial GPU mem %f",get_process_gpu_mem(false, false));
    image_init();


    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    std::ostringstream oss, hdr;

    // Print header
    hdr       << std::setw(30) << "Name"
              << std::setw(8) << "Threads"
              << std::setw(12) << "FPS"
              << std::setw(12) << "FPS(@720p)"
              << std::setw(18) << "Hash"
              << std::endl;

    int num_tests=sizeof(tests)/sizeof(test_t);
    for (int t=0;t<num_tests;t++) {
        const char *filename = tests[t].video_file;
        int tc=tests[t].num_threads;

        bool is_h265=strcmp(filename + strlen(filename) - strlen(".265"), ".265") == 0;

        // Launch threads
        std::vector<std::thread> workers;
        std::vector<ThreadResult> results(tc);

        auto start_mem = get_process_gpu_mem(false, false);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < tc; ++i) {
            workers.emplace_back(do_decode, filename, is_h265, &results[i], tests[t].num_iters, tests[t].compute_hash);
        }
        auto end = std::chrono::high_resolution_clock::now();
        for (auto &t : workers) t.join();

        auto end_mem = get_process_gpu_mem(false, false);

        // Verify hashes match
        uint64_t ref_hash = results[0].total_hash;
        bool all_match = true;
        double total_fps = 0;
        double total_mbps = 0;
        for (auto &r : results) {
            total_fps += r.fps;
            total_mbps +=r.mbps;
            if (r.total_hash != ref_hash) all_match = false;
        }

        // Print
        oss         << std::setw(30) << tests[t].name
                    << std::setw(8) << tc;

        if (results[0].compute_hash)
        {
            oss << std::setw(12) << "N/A" << std::setw(12) << "N/A";
            oss << "        " << (all_match ? "OK" : "MISMATCH") << ":0x"
                << std::hex << ref_hash << std::dec;
        } else {
            oss << std::setw(10) << std::fixed << std::setw(12) << (int)total_fps << std::setw(12) << (int)(total_mbps/3600.0);
            oss << std::setw(18) << "N/A";
        }
        oss << "\n";

        std::cout << hdr.str() << oss.str();
        allocation_tracker_reset();
    }

    return 0;
}
