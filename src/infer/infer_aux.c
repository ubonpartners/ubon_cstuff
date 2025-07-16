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
#include <math.h>
#include "infer_aux.h"
#include "solvers.h"
#include "trt_stuff.h"
#include <cuda_fp16.h> // for __half

using namespace nvinfer1;

struct infer_aux {
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* ctx;
    cudaStream_t stream;
    aux_model_description_t md;
};

enum { TRT_OUTPUT_BUFFER_COUNT=1 };

infer_aux_t* infer_aux_create(const char* model_trt, const char *config_yaml) {
    trt_init();

    // Load serialized engine
    FILE* f = fopen(model_trt, "rb");
    if (!f) { perror("fopen"); return nullptr; }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    rewind(f);
    void* data = malloc(sz);
    assert(sz==fread(data, 1, sz, f));
    fclose(f);

    infer_aux_t* inf = (infer_aux_t*)calloc(1, sizeof(*inf));
    inf->runtime = createInferRuntime(trt_Logger);
    assert(inf->runtime!=0);
    inf->runtime->setGpuAllocator(&trt_allocator);
    inf->engine  = inf->runtime->deserializeCudaEngine(data, sz);
    free(data);
    assert(inf->engine);

    IEngineInspector* inspector = inf->engine->createEngineInspector();
    if (inspector)
    {
        const char* engineInfo = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
        inf->md.engineInfo=strdup(engineInfo);
        delete inspector;
    }

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
            inf->md.input_h    = dims.d[2];
            inf->md.input_w    = dims.d[3];
            inf->md.input_fp16 = (dt == DataType::kHALF);

            Dims dims_max=inf->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            assert(dims_max.nbDims==4);
            inf->md.max_batch=dims_max.d[0];

        } else {
            // assume output is [N,C]
            DataType dt       = inf->engine->getTensorDataType(name);
            inf->md.output_fp16 = (dt == DataType::kHALF);
            inf->md.embedding_size = dims.d[1];
        }
    }

    log_info("Loaded embedding model input_fp16=%d max_batch=%d input %dx%d emb %d",inf->md.input_fp16, inf->md.max_batch, inf->md.input_w,inf->md.input_h, inf->md.embedding_size );

    return inf;
}

void infer_aux_destroy(infer_aux_t* inf) {
    if (!inf) return;
    cudaStreamDestroy(inf->stream);
    if (inf->ctx) delete inf->ctx;
    if (inf->engine) delete inf->engine;
    if (inf->runtime) delete inf->runtime;
    if (inf->md.engineInfo) free((void*)inf->md.engineInfo);
    free(inf);
}

static float *infer_aux_batch_affine(infer_aux_t* inf, image_t** img, float *M, int n)
{
    // do inference on images given already computed affine warp parameters M
    // M should have should be 6 floats per batch entry

    // allocate device input
    size_t plane = inf->md.input_w * inf->md.input_h;
    size_t dtype = inf->md.input_fp16 ? 2 : 4;
    void* d_in;
    cudaMalloc(&d_in, n * 3 * plane * dtype);

    image_t *img_yuv[n];
    for(int i=0;i<n;i++) image_check(img[i]);
    for(int i=0;i<n;i++) img_yuv[i]=image_convert(img[i], IMAGE_FORMAT_YUV420_DEVICE);

    const image_t **img_const = (const image_t**)img_yuv;

    cuda_warp_yuv420_to_planar_float(
        img_const, d_in, n,
        inf->md.input_w, inf->md.input_h,
        M, false, inf->md.input_fp16,
        inf->stream
    );

    // Allocate output buffer
    size_t total = n * inf->md.embedding_size;
    float* h_out = (float*)malloc(total * sizeof(float));

    void* d_out;
    size_t d_out_bytes = total * (inf->md.output_fp16 ? sizeof(__half) : sizeof(float));
    cudaMalloc(&d_out, d_out_bytes);

    // TensorRT IO binding
    int nbIO = inf->engine->getNbIOTensors();
    for (int i = 0; i < nbIO; ++i) {
        const char* name = inf->engine->getIOTensorName(i);
        if (inf->engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            inf->ctx->setInputShape(name, Dims4{n, 3, inf->md.input_h, inf->md.input_w});
            inf->ctx->setTensorAddress(name, d_in);
        } else {
            inf->ctx->setTensorAddress(name, d_out);
        }
    }

    // Run inference
    inf->ctx->enqueueV3(inf->stream);

    // Copy and convert output
    if (inf->md.output_fp16) {
        __half* h_fp16 = (__half*)malloc(total * sizeof(__half));
        cudaMemcpyAsync(h_fp16, d_out, total * sizeof(__half), cudaMemcpyDeviceToHost, inf->stream);
        cudaStreamSynchronize(inf->stream);
        for (size_t i = 0; i < total; ++i) {
            h_out[i] = __half2float(h_fp16[i]);
        }
        free(h_fp16);
    } else {
        cudaMemcpyAsync(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost, inf->stream);
        cudaStreamSynchronize(inf->stream);
    }

    int sz=inf->md.embedding_size;
    if (sz!=360) // MDB:HACK FIXME
    {
        for(int i=0;i<n;i++)
        {
            float norm=0;
            int sz=inf->md.embedding_size;
            for(int j=0;j<sz;j++) norm+=(h_out[i*sz+j]*h_out[i*sz+j]);
            if (norm>0.0f)
            {
                float scale=1.0/sqrtf(norm);
                for(int j=0;j<sz;j++) h_out[i*sz+j]*=scale;
            }
        }
    }

    // Cleanup
    cudaFree(d_out);
    cudaFree(d_in);
    for(int i=0;i<n;i++) destroy_image(img_yuv[i]);
    return h_out;
}

float* infer_aux_batch(infer_aux_t* inf, image_t** img, float* kp, int n) {
    assert(n <= inf->md.max_batch);

    // affine transform
    float* M = (float*)malloc(n * 6 * sizeof(float));
    solve_affine_face_points(img, kp, n, inf->md.input_w, inf->md.input_h, M);

    float *ret=infer_aux_batch_affine(inf, img, M, n);

    free(M);
    return ret;
}

void infer_aux_batch(infer_aux_t *inf, image_t **img, embedding_t **ret_emb, float *kp, int n)
{
    float *p=infer_aux_batch(inf, img, kp, n);
    int sz=(int)inf->md.embedding_size;
    for(int i=0;i<n;i++)
    {
        embedding_check(ret_emb[i]);
        embedding_set_data(ret_emb[i], p+sz*i, sz);
    }
    free(p);
}

void infer_aux_batch_roi(infer_aux_t *inf, image_t **img, embedding_t **ret_emb, roi_t *rois, int n)
{
    assert(n <= inf->md.max_batch);

    float* M = (float*)malloc(n * 6 * sizeof(float));
    solve_affine_points_roi(img, rois, n, inf->md.input_w, inf->md.input_h, M);

    float *p=infer_aux_batch_affine(inf, img, M, n);

    free(M);

    int sz=(int)inf->md.embedding_size;
    for(int i=0;i<n;i++)
    {
        embedding_check(ret_emb[i]);
        embedding_set_data(ret_emb[i], p+sz*i, sz);
    }

    free(p);
}

aux_model_description_t *infer_aux_get_model_description(infer_aux_t *inf)
{
    return &inf->md;
}