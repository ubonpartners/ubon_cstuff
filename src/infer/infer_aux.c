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
#include "display.h"
#include "maths_stuff.h"

using namespace nvinfer1;

#define MAX_OUTPUTS 2

struct infer_aux {
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* ctx;
    cudaStream_t stream;
    bool l2_normalize[MAX_OUTPUTS];
    int output_size[MAX_OUTPUTS];
    bool output_fp16[MAX_OUTPUTS];
    void *output_mem_device[MAX_OUTPUTS];
    void *output_mem_host[MAX_OUTPUTS];
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
    int noutput=0;
    for (int i = 0; i < nbIO; ++i) {
        const char* name = inf->engine->getIOTensorName(i);
        TensorIOMode mode = inf->engine->getTensorIOMode(name);
        Dims dims         = inf->engine->getTensorShape(name);
        DataType dt       = inf->engine->getTensorDataType(name);

        if (mode == TensorIOMode::kINPUT) {
            Dims dims_max=inf->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            if (dims.nbDims == 4)
            {
                inf->md.input_ch   = dims.d[1];
                inf->md.input_h    = dims.d[2];
                inf->md.input_w    = dims.d[3];
            }
            else if (dims.nbDims==3)
            {
                inf->md.input_ch   = 1;//dims.d[0];
                inf->md.input_h    = dims.d[1];
                inf->md.input_w    = dims.d[2];
                log_debug("Dim3 input %s: %dx%dx%d",model_trt,inf->md.input_ch, inf->md.input_h, inf->md.input_w);
            }
            else
            {
                assert(0);
            }
            inf->md.input_dims=dims.nbDims;
            inf->md.input_fp16 = (dt == DataType::kHALF);
            inf->md.max_batch=dims_max.d[0];
            inf->md.max_w=dims_max.d[dims.nbDims-1];
            inf->md.max_h=dims_max.d[dims.nbDims-2];

        } else {
            // assume output is [N,C]
            DataType dt = inf->engine->getTensorDataType(name);
            for(int i=0;i<dims.nbDims;i++)
            {
                log_debug("output %s dim %d : %d",name,i,(int)dims.d[i]);
            }
            log_debug("output %s is_fp16 %d",name,(dt == DataType::kHALF));
            if (noutput==0)
            {

                inf->md.embedding_size = dims.d[dims.nbDims-1];
                inf->md.output_fp16 = (dt == DataType::kHALF);
            }
            else
            {
                inf->md.embedding_size_2nd_output= dims.d[dims.nbDims-1];
                inf->md.output2_fp16 = (dt == DataType::kHALF);
            }
            noutput++;
        }
    }
    log_debug("output embedding size %d max batch %d",inf->md.embedding_size,inf->md.max_batch);
    inf->output_size[0]=inf->md.embedding_size;
    inf->output_size[1]=inf->md.embedding_size_2nd_output;
    inf->output_fp16[0]=inf->md.output_fp16;
    inf->output_fp16[1]=inf->md.output2_fp16;
    for(int i=0;i<MAX_OUTPUTS;i++)
    {
        if (inf->output_size[i])
        {
            int output_mem_size=inf->md.max_batch*inf->output_size[i]*(inf->output_fp16[i] ? 2 : 4);
            log_debug("Allocating %d bytes of output memory",output_mem_size);
            output_mem_size=(output_mem_size+127)&(~127);
            inf->output_mem_device[i]=cuda_malloc(output_mem_size);
            inf->output_mem_host[i]=cuda_malloc_host(output_mem_size);
            inf->l2_normalize[i]=(inf->output_size[i]==512); // HACK FIXME
        }
    }

    log_info("Loaded embedding model input_fp16=%d output_fp16=%d max_batch=%d input %dx%d (max %dx%d) emb %d",
        inf->md.input_fp16, inf->output_fp16[0], inf->md.max_batch,
        inf->md.input_w,inf->md.input_h,
        inf->md.max_w, inf->md.max_h,
        inf->md.embedding_size );

    return inf;
}

void infer_aux_destroy(infer_aux_t* inf) {
    if (!inf) return;
    cudaStreamDestroy(inf->stream);
    for(int i=0;i<2;i++)
    {
        if (inf->output_mem_device[i]) cuda_free(inf->output_mem_device[i]);
        if (inf->output_mem_host[i]) cuda_free_host(inf->output_mem_host[i]);
    }
    if (inf->ctx) delete inf->ctx;
    if (inf->engine) delete inf->engine;
    if (inf->runtime) delete inf->runtime;
    if (inf->md.engineInfo) free((void*)inf->md.engineInfo);
    free(inf);
}

static void do_inference(infer_aux_t* inf, void *src, embedding_t **ret_emb, int n,int h, int w)
{
    int out_elements[MAX_OUTPUTS];
    int out_bytes[MAX_OUTPUTS];

    assert(w<=inf->md.max_w);
    assert(h<=inf->md.max_h);
    assert(n<=inf->md.max_batch);
    assert(src!=0);

    // TensorRT IO binding
    int nout=0;
    int nbIO = inf->engine->getNbIOTensors();
    for (int i = 0; i < nbIO; ++i) {
        const char* name = inf->engine->getIOTensorName(i);
        if (h==0) h=inf->md.input_h;
        if (w==0) w=inf->md.input_w;
        if (inf->engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            if (inf->md.input_dims==4)
                inf->ctx->setInputShape(name, Dims4{n, inf->md.input_ch, h, w});
            else
            {
                log_debug("Set shape %dx%dx%d",n,h,w);
                inf->ctx->setInputShape(name, Dims3{n, h, w});
            }
            inf->ctx->setTensorAddress(name, src);

        } else {
            out_elements[nout]=inf->output_size[nout];
            out_bytes[nout]=inf->output_size[nout]*(inf->output_fp16[nout] ? 2 : 4);
            inf->ctx->setTensorAddress(name, inf->output_mem_device[nout++]);
            assert(nout<=MAX_OUTPUTS);
        }
    }

    // Run inference
    inf->ctx->enqueueV3(inf->stream);
    CHECK_CUDART_CALL(cudaStreamSynchronize(inf->stream));

    for(int i=0;i<nout;i++)
    {
        CHECK_CUDART_CALL(cudaMemcpyAsync(inf->output_mem_host[i],inf->output_mem_device[i],out_bytes[i]*n, cudaMemcpyDeviceToHost, inf->stream));
    }
    CHECK_CUDART_CALL(cudaStreamSynchronize(inf->stream));

    for(int i=0;i<nout;i++)
    {
        if (i!=0) continue; // for now ignore other outputs
        uint8_t *out=(uint8_t *)inf->output_mem_host[i];
        for (int b=0;b<n;b++)
        {
            if (inf->output_fp16[i])
                embedding_set_data_half(ret_emb[b], out, out_elements[i], inf->l2_normalize[i]);
            else
                embedding_set_data(ret_emb[b], (float *)out, out_elements[i], inf->l2_normalize[i]);
            out+=out_bytes[i];
        }
    }
}

void infer_aux_batch_tensor(infer_aux_t* inf, image_t **images, embedding_t **ret_emb, int n)
{
    assert(n <= inf->md.max_batch);
    image_t *converted_images[n];
    image_format_t fmt=(inf->md.input_fp16) ? IMAGE_FORMAT_TENSOR_FP16_DEVICE : IMAGE_FORMAT_TENSOR_FP32_DEVICE;
    for(int i=0;i<n;i++)
    {
        image_t *img=images[i];
        assert(img->width==inf->md.input_w || inf->md.input_w==-1);
        assert(img->height==inf->md.input_h || inf->md.input_h==-1);
        assert(img->c==1);
        converted_images[i]=image_convert(img, fmt);
    }

    for(int i=0;i<n;i++)
    {
        cuda_stream_add_dependency(inf->stream, converted_images[i]->stream);
        do_inference(inf, converted_images[i]->tensor_mem, ret_emb+i, 1, images[0]->height, images[0]->width);
    }
    for(int i=0;i<n;i++) image_destroy(converted_images[i]);
}

static void infer_aux_batch_affine(infer_aux_t* inf, image_t** img, embedding_t **ret_emb, float *M, int n)
{
    // do inference on images given already computed affine warp parameters M
    // M should have should be 6 floats per batch entry

    // allocate device input
    size_t plane = inf->md.input_w * inf->md.input_h;
    size_t dtype = inf->md.input_fp16 ? 2 : 4;

    //int elt_size=(fmt==IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE) ? 2 : 4;
    image_t *inf_image=image_create(inf->md.input_w, inf->md.input_h*n,
        inf->md.input_fp16 ? IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE: IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE);

    image_t *img_yuv[n];
    for(int i=0;i<n;i++) image_check(img[i]);
    for(int i=0;i<n;i++) img_yuv[i]=image_convert(img[i], IMAGE_FORMAT_YUV420_DEVICE);

    const image_t **img_const = (const image_t**)img_yuv;

    cuda_warp_yuv420_to_planar_float(
        img_const, inf_image->rgb, n,
        inf->md.input_w, inf->md.input_h,
        M, false, inf->md.input_fp16,
        inf->stream
    );

    do_inference(inf, inf_image->rgb, ret_emb, n, 0, 0);

    // Cleanup
    image_destroy(inf_image);
    for(int i=0;i<n;i++) image_destroy(img_yuv[i]);
}

float* infer_aux_batch(infer_aux_t* inf, image_t** img, float* kp, int n) {
    assert(n <= inf->md.max_batch);
    float M[n*6];
    solve_affine_face_points(img, kp, n, inf->md.input_w, inf->md.input_h, M);
    embedding_t *e=embedding_create(inf->output_size[0], 0);
    infer_aux_batch_affine(inf, img, &e, M, n);
    float *ret=(float *)malloc(sizeof(float)*inf->output_size[0]);
    memcpy(ret, embedding_get_data(e), sizeof(float)*inf->output_size[0]);
    embedding_destroy(e);
    return ret;
}

void infer_aux_batch(infer_aux_t *inf, image_t **img, embedding_t **ret_emb, float *kp, int n)
{
    assert(n <= inf->md.max_batch);
    float M[n*6];
    solve_affine_face_points(img, kp, n, inf->md.input_w, inf->md.input_h, M);
    infer_aux_batch_affine(inf, img, ret_emb, M, n);
}

void infer_aux_batch_roi(infer_aux_t *inf, image_t **img, embedding_t **ret_emb, roi_t *rois, int n)
{
    assert(n <= inf->md.max_batch);
    float M[n*6];
    solve_affine_points_roi(img, rois, n, inf->md.input_w, inf->md.input_h, M);
    infer_aux_batch_affine(inf, img, ret_emb, M, n);
}

aux_model_description_t *infer_aux_get_model_description(infer_aux_t *inf)
{
    return &inf->md;
}