#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cassert>
#include <memory>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include "NvInfer.h"
#include "image.h"
#include "infer.h"
#include "NvOnnxParser.h"
#include "display.h"
#include "detections.h"
#include "cuda_stuff.h"
#include "log.h"
#include <yaml-cpp/yaml.h>

using namespace nvinfer1;
using namespace nvonnxparser;

struct infer
{
    IRuntime *runtime;
    ICudaEngine* engine;
    IExecutionContext *ec;
    CUdeviceptr output_mem;
    cudaStream_t stream;
    void *output_mem_host;
    int output_size;
    int fn;
    int detection_attribute_size;
    const char *input_tensor_name;
    const char *output_tensor_name;
}; 

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


static void infer_build(const char *onnx_filename, const char *out_filename)
{
    const char *tc_filename="trt_timing_cache.dat";
    
    auto builder = nvinfer1::createInferBuilder(trt_Logger);
    assert(builder!=0);

    auto network = builder->createNetworkV2(0);
    assert(network!=0);

    auto config=builder->createBuilderConfig();
    assert(config!=0);

    auto parser=createParser(*network, trt_Logger);
    assert(parser!=0);

    config->clearFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
    config->setFlag(BuilderFlag::kFP16);

    // Check if the timing cache file exists
    std::unique_ptr<nvinfer1::ITimingCache> timingCache;
    if (std::filesystem::exists(tc_filename)) 
    {
        // Load the existing timing cache from file
        std::ifstream cacheFile(tc_filename, std::ios::binary);
        if (cacheFile) 
        {
            std::vector<char> buffer(std::istreambuf_iterator<char>(cacheFile), {});
            timingCache.reset(config->createTimingCache(buffer.data(), buffer.size()));
            cacheFile.close();
        }
    } 
    else 
    {
        // Create a new timing cache if the file does not exist
        timingCache.reset(config->createTimingCache(nullptr, 0));
    }
    // Set the timing cache in the builder config
    config->setTimingCache(*timingCache, false);

    auto parsed = parser->parseFromFile(onnx_filename, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    assert(parsed!=0);

    for (int i = 0; i < network->getNbInputs(); ++i) 
    {
        ITensor* inputTensor = network->getInput(i);
        inputTensor->setType(DataType::kHALF);
        assert(inputTensor->getType()==DataType::kHALF);
        Dims inputDims = inputTensor->getDimensions();
        for (int j = 0; j < inputDims.nbDims; ++j) 
        {
            log_debug("Input (%s) %d : %d\n",inputTensor->getName(), i, (int)inputDims.d[j]);
        }
    }

    IOptimizationProfile* profile = builder->createOptimizationProfile();

    ITensor* inputTensor = network->getInput(0);
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 64, 64});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 640, 352});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims4{16, 3, 640, 640});
    config->addOptimizationProfile(profile);

    auto out=builder->buildSerializedNetwork(*network, *config);
    assert(out!=0);
    if (1)
    {
        FILE *fo=fopen(out_filename, "wb");
        if (fo)
        {
            fwrite(out->data(), 1, out->size(), fo);
            fclose(fo);
        }
    }

    // Save the updated timing cache to file if the build was successful
    if (out) 
    {
        const nvinfer1::ITimingCache* updatedTimingCache = config->getTimingCache();
        if (updatedTimingCache) 
        {
            std::ofstream cacheFile(tc_filename, std::ios::binary);
            if (cacheFile) 
            {
                nvinfer1::IHostMemory *serializedCache = updatedTimingCache->serialize();
                cacheFile.write((const char *)serializedCache->data(), serializedCache->size());
                cacheFile.close();
            }
        }
    }
}

infer_t *infer_create(const char *model, const char *yaml_config)
{
    infer_t *inf=(infer_t *)malloc(sizeof(infer_t));
    memset(inf, 0, sizeof(infer_t));

    cudaStreamCreate(&inf->stream);

    log_debug("Infer create");

    /*YAML::Node config = YAML::LoadFile(yaml_config);
    // Access 'kpt_shape'
    std::vector<int> kpt_shape = config["dataset"]["kpt_shape"].as<std::vector<int>>();
    std::cout << "kpt_shape: ";
    for (int v : kpt_shape) std::cout << v << " ";
    std::cout << "\n";

    // Access 'names'
    std::vector<std::string> names = config["dataset"]["names"].as<std::vector<std::string>>();
    std::cout << "names: ";
    for (const auto& name : names) std::cout << name << " ";
    std::cout << "\n";*/

    //infer_build("/mldata/weights/onnx/yolo11l-dpa-131224.onnx", "/mldata/weights/trt/yolo11l-dpa-131224.trt");

    FILE* model_file = fopen(model, "rb");
    assert(model_file!=0);
    fseek(model_file, 0, SEEK_END);
    long model_size = ftell(model_file);
    fseek(model_file, 0, SEEK_SET);
    uint8_t* engineData = (uint8_t *)malloc(model_size);
    assert(model_size==fread(engineData, 1, model_size, model_file));
    fclose(model_file);

    inf->runtime = createInferRuntime(trt_Logger);
    assert(inf->runtime!=0);

    inf->engine = inf->runtime->deserializeCudaEngine(engineData, model_size);
    assert(inf->engine!=0);
    free(engineData);

   // int detection_attribute_size = inf->engine->getBindingDimensions(1).d[1];
  //  printf("detection_attribute_size=%d\n", detection_attribute_size);

    inf->ec = inf->engine->createExecutionContext(ExecutionContextAllocationStrategy::kSTATIC);
    assert(inf->ec!=0);
    //inf->ec->setOptimizationProfileAsync(0, 0);

    int num_bindings=inf->engine->getNbIOTensors();
    int num_input=0;
    int num_output=0;
    for(int i=0;i<num_bindings;i++)
    {
        const char* name = inf->engine->getIOTensorName(i);
        TensorIOMode iomode=inf->engine->getTensorIOMode(name);
        if (iomode==nvinfer1::TensorIOMode::kINPUT)
        {
            assert(num_input==0);
            num_input++;
            inf->input_tensor_name=name;
        }
        else
        {
            assert(num_output==0);
            num_output++;
            inf->output_tensor_name=name;
            Dims outputDims=inf->engine->getTensorShape(inf->output_tensor_name);
            inf->detection_attribute_size=(int)outputDims.d[1];
            log_debug("outputDims %dx%dx%d\n", (int)outputDims.d[0], (int)outputDims.d[1], (int)outputDims.d[2]);
        }
    }
    return inf;
}

void infer_destroy(infer_t *inf)
{
    if (!inf) return;
    cudaStreamDestroy(inf->stream);
    if (inf->ec) delete inf->ec;
    if (inf->engine) delete inf->engine;
    if (inf->runtime) delete inf->runtime;
    if (inf->output_mem) cuMemFree(inf->output_mem);
    inf->output_mem=0;
    if (inf->output_mem_host) cuMemFreeHost(inf->output_mem_host);
    inf->output_mem_host=0;
    free(inf);
}

detections_t *infer(infer_t *inf, image_t *img)
{
    image_t *img_device=image_convert(img, IMAGE_FORMAT_YUV420_DEVICE);
    image_t *image_scaled=image_scale(img_device, 640, 384);
    assert(image_scaled!=0);

    image_t *image_input=image_convert(image_scaled, IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE);
    assert(image_input!=0);

    auto input_dims = nvinfer1::Dims4{1, 3, image_input->height, image_input->width};
    assert(true==inf->ec->setInputShape(inf->input_tensor_name, input_dims));
    assert(true==inf->ec->setTensorAddress(inf->input_tensor_name, image_input->rgb));

    int max_output_size=(int)inf->ec->getMaxOutputSize(inf->output_tensor_name);
    if (max_output_size>inf->output_size)
    {
        log_debug("Reallocate output memory [%d bytes]\n",max_output_size);
        if (inf->output_mem) cuMemFree(inf->output_mem);
        if (inf->output_mem_host) cuMemFreeHost(inf->output_mem_host);
        cuMemAlloc(&inf->output_mem, max_output_size);
        cuMemAllocHost(&inf->output_mem_host, max_output_size);
        inf->output_size=max_output_size;
    }

    assert(true==inf->ec->setTensorAddress(inf->output_tensor_name, (void*)inf->output_mem));
    assert(true==inf->ec->allInputDimensionsSpecified());
    assert(0==inf->ec->inferShapes(0,0));

    Dims outputDims = inf->ec->getTensorShape(inf->output_tensor_name);
    int columns=outputDims.d[1];
    int rows=outputDims.d[2];

    cuda_stream_add_dependency(inf->stream, image_input->stream);
    assert(true==inf->ec->enqueueV3(inf->stream));
    cudaMemcpyAsync(inf->output_mem_host, (void*)inf->output_mem, inf->output_size, cudaMemcpyDeviceToHost,inf->stream);
    cuStreamSynchronize(inf->stream);

    detections_t *dets=create_detections(300);

    float *p=(float *)inf->output_mem_host;
    float probsum=0;
    
    int num_classes=5;
    for(int i=0;i<rows;i++)
    {
        float max_prob=0;
        int best_cl=0;
        for(int j=0;j<num_classes;j++)
        {
            float prob=p[(4+j)*rows+i];
            if (prob>max_prob)
            {
                max_prob=prob;
                best_cl=j;
            }
        } 
        probsum+=max_prob; 
        if (max_prob>0.4)
        {
            detection_t *det=detection_add_end(dets);
            if (det)
            {
                float cx=p[0*rows+i];
                float cy=p[1*rows+i];
                float w=p[2*rows+i];
                float h=p[3*rows+i];

                det->x0=cx-w*0.5;
                det->x1=cx+w*0.5;
                det->y0=cy-h*0.5;
                det->y1=cy+h*0.5;
                det->index=i;
                det->cl=best_cl;
                det->conf=max_prob;
            }
        } 
    }

    detections_nms_inplace(dets, 0.5);
    //show_detections(dets);

    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *d=&dets->det[i];
        detection_create_keypoints(d);
        keypoints_t *kp=d->kp;
        int index=d->index;
        for(int i=0;i<5;i++)
        {
            kp->face_points[i].x=p[(4+num_classes+3*i+0)*rows+index];
            kp->face_points[i].y=p[(4+num_classes+3*i+1)*rows+index];
            kp->face_points[i].conf=p[(4+num_classes+3*i+2)*rows+index];
        }
        for(int i=0;i<17;i++)
        {
            kp->pose_points[i].x=p[(4+num_classes+3*(i+5)+0)*rows+index];
            kp->pose_points[i].y=p[(4+num_classes+3*(i+5)+1)*rows+index];
            kp->pose_points[i].conf=p[(4+num_classes+3*(i+5)+2)*rows+index];
        }
    }
    detections_scale(dets, 1.0/image_input->width, 1.0/image_input->height);

    inf->fn++;
    destroy_image(img_device);
    destroy_image(image_scaled);
    destroy_image(image_input);

    return dets;
}