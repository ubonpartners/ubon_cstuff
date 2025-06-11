#include <math.h>
#include "image.h"
#include "cuda_stuff.h"
#include "cuda_kernels.h"       // cuda_warp_yuv420_to_planar_float
#include "cuda_runtime.h"
#include "NvInfer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "infer_aux.h"
#include "solvers.h"

using namespace nvinfer1;

struct infer_aux {
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* ctx;
    cudaStream_t stream;
    bool input_fp16;
    int max_batch;
    int input_w;
    int input_h;
    int embedding_size;
};

enum { TRT_OUTPUT_BUFFER_COUNT=1 };

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept
    {
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
            log_error("[TRT] %s\n",msg);
        if (severity == Severity::kWARNING)
            log_warn("[TRT] %s\n",msg);
    }
} trt_Logger;

infer_aux_t* infer_aux_create(const char* model_trt) {
    // Load serialized engine
    FILE* f = fopen(model_trt, "rb");
    if (!f) { perror("fopen"); return nullptr; }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    rewind(f);
    void* data = malloc(sz);
    fread(data, 1, sz, f);
    fclose(f);

    infer_aux_t* inf = (infer_aux_t*)calloc(1, sizeof(*inf));
    inf->runtime = createInferRuntime(trt_Logger);
    inf->engine  = inf->runtime->deserializeCudaEngine(data, sz);
    free(data);
    assert(inf->engine);

    inf->ctx = inf->engine->createExecutionContext();
    cudaStreamCreate(&inf->stream);

    // Inspect IO tensors via modern API
    int nbIO = inf->engine->getNbIOTensors();
    for (int i = 0; i < nbIO; ++i) {
        const char* name = inf->engine->getIOTensorName(i);
        TensorIOMode mode = inf->engine->getTensorIOMode(name);
        Dims dims         = inf->engine->getTensorShape(name);
        DataType dt       = inf->engine->getTensorDataType(name);

        if (mode == TensorIOMode::kINPUT) {
            assert(dims.nbDims == 4);
            inf->input_h    = dims.d[2];
            inf->input_w    = dims.d[3];
            inf->input_fp16 = (dt == DataType::kHALF);

            Dims dims_max=inf->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            assert(dims_max.nbDims==4);
            inf->max_batch=dims_max.d[0];

        } else {
            // assume output is [N,C]
            inf->embedding_size = dims.d[1];
        }
    }

    log_info("Loaded embedding model input_fp16=%d max_batch=%d input %dx%d emb %d",inf->input_fp16, inf->max_batch, inf->input_w,inf->input_h, inf->embedding_size );

    return inf;
}

void infer_aux_destroy(infer_aux_t* inf) {
    if (!inf) return;
    cudaStreamDestroy(inf->stream);
    if (inf->ctx) delete inf->ctx;
    if (inf->engine) delete inf->engine;
    if (inf->runtime) delete inf->runtime;
    free(inf);
}

float* infer_aux_batch(infer_aux_t* inf, image_t** img, float* kp, int n) {
    assert(n <= inf->max_batch);

    // allocate device input
    size_t plane = inf->input_w * inf->input_h;
    size_t dtype = inf->input_fp16 ? 2 : 4;
    void* d_in;
    cudaMalloc(&d_in, n * 3 * plane * dtype);

    float* M = (float*)malloc(n * 6 * sizeof(float));
    solve_affine_face_points(img, kp, n, inf->input_w, inf->input_h, M);

    // warp (cast to const)
    const image_t** img_const = (const image_t**)img;
    cuda_warp_yuv420_to_planar_float(
        img_const, d_in, n,
        inf->input_w, inf->input_h,
        M, false, inf->input_fp16,
        inf->stream
    );
    free(M);

    // prepare output
    float* h_out = (float*)malloc(n * inf->embedding_size * sizeof(float));
    void*  d_out;
    cudaMalloc(&d_out, n * inf->embedding_size * sizeof(float));

    // set addresses & shapes
    int nbIO = inf->engine->getNbIOTensors();
    for (int i = 0; i < nbIO; ++i) {
        const char* name = inf->engine->getIOTensorName(i);
        if (inf->engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            inf->ctx->setInputShape(name, Dims4{n, 3, inf->input_h, inf->input_w});
            inf->ctx->setTensorAddress(name, d_in);
        } else {
            inf->ctx->setTensorAddress(name, d_out);
        }
    }

    // run
    inf->ctx->enqueueV3(inf->stream);

    // copy back
    cudaMemcpyAsync(h_out, d_out,
                    n * inf->embedding_size * sizeof(float),
                    cudaMemcpyDeviceToHost, inf->stream);
    cudaStreamSynchronize(inf->stream);

    // cleanup
    cudaFree(d_out);
    cudaFree(d_in);

    return h_out;
}