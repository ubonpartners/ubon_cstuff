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
#include <cuda_runtime_api.h>
#include <unistd.h>
#include "NvInfer.h"
#include "image.h"
#include "infer.h"
#include "NvOnnxParser.h"
#include "display.h"
#include "detections.h"
#include "cuda_stuff.h"
#include "cuda_kernels.h"
#include "log.h"
#include "cuda_nms.h"
#include "yaml_stuff.h"
#include "trt_stuff.h"

#define debugf if (0) printf

using namespace nvinfer1;
using namespace nvonnxparser;

static const char *default_config="/mldata/config/train/train_yolo_dpa_l.yaml";

struct infer
{
    IRuntime *runtime;
    ICudaEngine* engine;
    IExecutionContext *ec;
    void *output_mem;
    cudaStream_t stream;
    CudaNMSHandle nms;
    int nms_max_boxes;
    void *output_mem_host;
    int person_class_index, face_class_index;
    int max_batch;
    int min_w, min_h;
    int max_w, max_h;
    bool use_cuda_nms;
    bool do_fuse_face_person;
    bool inf_allow_upscale;
    bool input_is_fp16, output_is_fp16;
    int output_size;
    int fn;
    float nms_thr;
    float det_thr;
    int inf_limit_min_width;
    int inf_limit_min_height;
    int inf_limit_max_width;
    int inf_limit_max_height;
    int inf_limit_max_batch;
    int max_detections;
    int detection_attribute_size;
    const char *input_tensor_name;
    const char *output_tensor_name;
    model_description_t md;
};

// Platform specific changes
static void create_exec_context(infer_t *inf)
{
#if (UBONCSTUFF_PLATFORM == 1) // Orin Nano
    inf->ec = inf->engine->createExecutionContext();
#elif (UBONCSTUFF_PLATFORM == 0) // Desktop Nvidia GPU
    inf->ec = inf->engine->createExecutionContext(ExecutionContextAllocationStrategy::kSTATIC);
#else
    #error "Unsupported Platform"
#endif

    return;
}

image_t *load_rgb32_tensor(const char *file, int w, int h)
{
    FILE *f=fopen(file, "rb");
    if (!f)
    {
        log_error("Could not open file %s",file);
        return 0;
    }
    image_t *img=create_image(w, h, IMAGE_FORMAT_RGB_PLANAR_FP32_HOST);
    assert(w*h*4*3==fread(img->rgb, 1, w*h*4*3, f));
    fclose(f);
    return img;
}

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

void infer_print_model_description(model_description_t *desc)
{
    if (!desc) {
        std::cerr << "Model description is null.\n";
        return;
    }

    std::cout << "=== Model Description ===\n";

    std::cout << "Model Output Dimensions: ["
              << desc->model_output_dims[0] << " x "
              << desc->model_output_dims[1] << " x "
              << desc->model_output_dims[2] << "]\n";

    std::cout << "Input Width Range: " << desc->min_w << " - " << desc->max_w << "\n";
    std::cout << "Input Height Range: " << desc->min_h << " - " << desc->max_h << "\n";

    std::cout << "Keypoints: " << desc->num_keypoints << "\n";
    std::cout << "Max Batch Size: " << desc->max_batch << "\n";
    std::cout << "Input Format: " << (desc->input_is_fp16 ? "FP16" : "FP32") << "\n";
    std::cout << "Output Format: " << (desc->output_is_fp16 ? "FP16" : "FP32") << "\n";
    std::cout << "REID vector length: " << desc->reid_vector_len << "\n";

    std::cout << "Classes (" << desc->num_classes << "):\n";
    for (size_t i = 0; i < desc->class_names.size(); ++i) {
        std::cout << "  [" << i << "] " << desc->class_names[i] << "\n";
    }

    std::cout << "Person Attributes (" << desc->num_person_attributes << "):\n";
    for (size_t i = 0; i < desc->person_attribute_names.size(); ++i) {
        std::cout << "  [" << i << "] " << desc->person_attribute_names[i] << "\n";
    }
}

model_description_t *infer_get_model_description(infer_t *inf)
{
    if (!inf) return 0;
    return &inf->md;
}

infer_t *infer_create(const char *model, const char *yaml_config)
{
    trt_init();

    infer_t *inf=(infer_t *)malloc(sizeof(infer_t));
    memset(inf, 0, sizeof(infer_t));

    inf->nms_thr=0.45;
    inf->det_thr=0.2;
    inf->inf_limit_max_width=640;
    inf->inf_limit_max_height=640;
    inf->inf_limit_min_width=128;
    inf->inf_limit_min_height=128;
    inf->inf_limit_max_batch=8;
    inf->max_detections=200;
    inf->person_class_index=-1;
    inf->face_class_index=-1;
    inf->use_cuda_nms=true;
    inf->inf_allow_upscale=false; // to be like ultralytics
    inf->do_fuse_face_person=true;

    cudaStreamCreate(&inf->stream);

    log_debug("Infer create");

    if ((yaml_config==0)||(strlen(yaml_config)==0))
    {
        log_info("infer_create: No config specified; trying default:");
        yaml_config=default_config;
    }

    YAML::Node config = yaml_load(yaml_config);
    std::vector<std::string> names = config["dataset"]["names"].as<std::vector<std::string>>();
    std::vector<int> kpt_shape = config["dataset"]["kpt_shape"].as<std::vector<int>>();
    model_description_t md = {};
    for (const auto& name : names)
    {
        if (name.size() >= 7 && name.compare(0, 7, "person_") == 0)
            md.person_attribute_names.push_back(name);
        else
        {
            if (name.size()==6 && name.compare(0, 6, "person")==0)
                inf->person_class_index=md.class_names.size();
            if (name.size()==4 && name.compare(0, 6, "face")==0)
                inf->face_class_index=md.class_names.size();
            md.class_names.push_back(name);

        }
    }
    md.num_classes=md.class_names.size();
    md.num_person_attributes=md.person_attribute_names.size();
    md.num_keypoints=kpt_shape[0];

    inf->md=md;
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
    inf->runtime->setGpuAllocator(&trt_allocator);

    inf->engine = inf->runtime->deserializeCudaEngine(engineData, model_size);
    assert(inf->engine!=0);
    free(engineData);

   // int detection_attribute_size = inf->engine->getBindingDimensions(1).d[1];
  //  printf("detection_attribute_size=%d\n", detection_attribute_size);

    create_exec_context(inf);
    assert(inf->ec!=0);
    //inf->ec->setOptimizationProfileAsync(0, 0);

    IEngineInspector* inspector = inf->engine->createEngineInspector();
    if (inspector)
    {
        const char* engineInfo = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
        inf->md.engineInfo=strdup(engineInfo);
        delete inspector;
    }

    int num_optimization_profiles=inf->engine->getNbOptimizationProfiles();
    assert(num_optimization_profiles==1);

    int num_bindings=inf->engine->getNbIOTensors();
    int num_input=0;
    int num_output=0;
    for(int i=0;i<num_bindings;i++)
    {
        const char* name = inf->engine->getIOTensorName(i);
        TensorIOMode iomode=inf->engine->getTensorIOMode(name);
        nvinfer1::DataType dtype = inf->engine->getTensorDataType(name);
        assert(dtype==nvinfer1::DataType::kFLOAT || dtype==nvinfer1::DataType::kHALF);

        if (iomode==nvinfer1::TensorIOMode::kINPUT)
        {
            assert(num_input==0);
            num_input++;
            inf->input_tensor_name=name;
            inf->md.input_is_fp16=inf->input_is_fp16=(dtype==nvinfer1::DataType::kHALF);

            Dims dims_max=inf->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            Dims dims_min=inf->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN);
            assert(dims_max.nbDims==4);
            assert(dims_min.nbDims==4);
            assert(dims_min.d[0]==1); // min batch should be 1!
            inf->md.max_batch=inf->max_batch=dims_max.d[0];
            inf->md.max_h=inf->max_h=dims_max.d[2];
            inf->md.max_w=inf->max_w=dims_max.d[3];
            inf->md.min_h=inf->min_h=dims_min.d[2];
            inf->md.min_w=inf->min_w=dims_min.d[3];
            log_debug("TRT model [%dx%d]->[%dx%d]; max batch %d",inf->min_w,inf->min_h,inf->max_w,inf->max_h,inf->max_batch);
        }
        else
        {
            assert(num_output==0);
            num_output++;
            inf->output_tensor_name=name;
            inf->md.output_is_fp16=inf->output_is_fp16=(dtype==nvinfer1::DataType::kHALF);
            Dims outputDims=inf->engine->getTensorShape(inf->output_tensor_name);
            inf->detection_attribute_size=(int)outputDims.d[1]; // "110"
            log_debug("outputDims %dx%dx%d\n", (int)outputDims.d[0], (int)outputDims.d[1], (int)outputDims.d[2]);
            inf->md.model_output_dims[0]=(int)outputDims.d[0];
            inf->md.model_output_dims[1]=(int)outputDims.d[1];
            inf->md.model_output_dims[2]=(int)outputDims.d[2];
        }
    }

    // verify that the number of outputs per-detection is as we expect based on the config
    int expected_size=4 // box
                      +inf->md.num_classes
                      +inf->md.num_person_attributes
                      +inf->md.num_keypoints*3;
    if (expected_size+64==inf->detection_attribute_size)
    {
        inf->md.reid_vector_len=64;
        log_debug("found REID vector length of %d\n",inf->md.reid_vector_len);
    }
    else if (expected_size!=inf->detection_attribute_size)
    {
        log_fatal("expected tensor size %d, got %d (%d classes, %d attr, %d kp)",
            expected_size,inf->detection_attribute_size,
            inf->md.num_classes,inf->md.num_person_attributes,inf->md.num_keypoints);
    }

    return inf;
}

void infer_destroy(infer_t *inf)
{
    log_debug("infer destroy %p",inf);
    if (!inf) return;
    CHECK_CUDART_CALL(cudaStreamDestroy(inf->stream));
    if (inf->nms) cuda_nms_free_workspace(inf->nms);
    if (inf->ec) delete inf->ec;
    if (inf->engine) delete inf->engine;
    if (inf->runtime) delete inf->runtime;
    cuda_free(inf->output_mem);
    inf->output_mem=0;
    cuda_free_host(inf->output_mem_host);
    inf->output_mem_host=0;
    if (inf->md.engineInfo) free((void*)inf->md.engineInfo);
    free(inf);
}

static detection_list_t *process_detections(infer_t *inf, float *p, int rows, int row_stride)
{
    detection_list_t *dets=detection_list_create(inf->max_detections); // output post-NMS detections

    int num_classes=inf->md.num_classes;
    int num_attributes=inf->md.num_person_attributes;
    int num_person_detections=0;
    int num_face_detections=0;
    for(int cl=0;cl<num_classes;cl++)
    {
        int num=0;
        for(int i=0;i<rows;i++)
        {
            float conf=p[(4+cl)*row_stride+i];
            if (conf>inf->det_thr) num++;
        }
        if (num!=0)
        {
            detection_list_t *temp_dets=detection_list_create(num); // will hold pre-NMS detections for 1 class
            for(int i=0;i<rows;i++)
            {
                float conf=p[(4+cl)*row_stride+i];
                if (conf>inf->det_thr)
                {
                    detection_t *det=detection_list_add_end(temp_dets);
                    assert(det!=0);
                    if (det)
                    {
                        float cx=p[0*row_stride+i];
                        float cy=p[1*row_stride+i];
                        float w=p[2*row_stride+i];
                        float h=p[3*row_stride+i];

                        det->x0=cx-w*0.5;
                        det->x1=cx+w*0.5;
                        det->y0=cy-h*0.5;
                        det->y1=cy+h*0.5;
                        det->index=i;
                        det->cl=cl;
                        det->conf=conf;
                        det->num_face_points=0;
                        det->num_pose_points=0;
                    }
                }
            }
            detection_list_nms_inplace(temp_dets, inf->nms_thr);
            for(int i=0;i<temp_dets->num_detections;i++) detection_list_append_copy(dets, temp_dets->det[i]);
            detection_list_destroy(temp_dets);
        }
    }


    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *d=dets->det[i];
        int index=d->index;
        if (d->cl==inf->face_class_index)
        {
            d->num_face_points=5;
            num_face_detections++;
            for(int i=0;i<d->num_face_points;i++)
            {
                d->face_points[i].x=p[(4+num_classes+num_attributes+3*i+0)*row_stride+index];
                d->face_points[i].y=p[(4+num_classes+num_attributes+3*i+1)*row_stride+index];
                d->face_points[i].conf=p[(4+num_classes+num_attributes+3*i+2)*row_stride+index];
            }
        }
        if (d->cl==inf->person_class_index)
        {
            num_person_detections++;
            d->num_pose_points=17;
            for(int i=0;i<d->num_pose_points;i++)
            {
                d->pose_points[i].x=p[(4+num_classes+num_attributes+3*(i+5)+0)*row_stride+index];
                d->pose_points[i].y=p[(4+num_classes+num_attributes+3*(i+5)+1)*row_stride+index];
                d->pose_points[i].conf=p[(4+num_classes+num_attributes+3*(i+5)+2)*row_stride+index];
            }
            d->num_attr=num_attributes;
            for(int i=0;i<num_attributes;i++)
            {
                d->attr[i]=p[(4+num_classes+i)*row_stride+index];
            }
            if (inf->md.reid_vector_len!=0)
            {
                d->reid_vector_len=inf->md.reid_vector_len;
                for(int i=0;i<inf->md.reid_vector_len;i++)
                {
                    d->reid[i]=p[(4+num_classes+num_attributes+3*(d->num_pose_points+d->num_face_points)+i)*row_stride+index];
                }
            }
        }
    }

    assert(inf->person_class_index==0); // no issue if not, just too lazy to fix code below!
    assert(inf->face_class_index==1);
    dets->person_dets=dets->det;
    dets->num_person_detections=num_person_detections;
    dets->face_dets=dets->person_dets+num_person_detections;
    dets->num_face_detections=num_face_detections;
    return dets;
}

static void process_detections_cuda_nms(infer_t *inf, int num, int columns, int numBoxes, detection_list_t **dets)
{
    if (numBoxes>inf->nms_max_boxes)
    {
        log_debug("increase cudaNMS boxes %d->%d",inf->nms_max_boxes,numBoxes);
        CHECK_CUDART_CALL(cudaStreamSynchronize(inf->stream));
        if (inf->nms) cuda_nms_free_workspace(inf->nms);
        inf->nms=0;
        inf->nms_max_boxes=0;
    }
    int max_detections_per_class=inf->max_detections;
    if (inf->nms==0)
    {
        inf->nms_max_boxes=std::max(1200, numBoxes);
        inf->nms=cuda_nms_allocate_workspace(inf->nms_max_boxes, inf->md.num_classes, max_detections_per_class, 0);
    }
    std::vector<std::vector<uint16_t>> keptIndices;
    std::vector<float> hostGathered;
    hostGathered.reserve(num * columns * 50); // roughly
    cuda_nms_run(
        inf->nms,
        (float *)inf->output_mem,
        num,
        numBoxes,
        inf->md.num_classes,
        columns,
        inf->det_thr,
        inf->nms_thr,
        max_detections_per_class,
        keptIndices,
        inf->stream);

    CHECK_CUDART_CALL(cudaStreamSynchronize(inf->stream));

    int nc=inf->md.num_classes;
    int num_attributes=inf->md.num_person_attributes;
    for(int b=0;b<num;b++)
    {
        hostGathered.clear();

        // Slice keptIndices for batch b
        std::vector<std::vector<uint16_t>> keptPerBatch;
        keptPerBatch.reserve(nc);
        for (int cl = 0; cl < nc; ++cl)
        {
            //printf("%d %d kept %d\n",b,cl,(int)keptIndices[b * nc + cl].size());
            keptPerBatch.push_back(keptIndices[b * nc + cl]);
        }

        cuda_nms_gather_kept_outputs(
            /*deviceOutputDev=*/ ((float*)inf->output_mem)+b*numBoxes*columns,
            /*numBoxes=*/       numBoxes,
            /*rowSize=*/        columns,
            /*keptIndices=*/    keptPerBatch,
            /*hostGathered=*/   hostGathered
        );

        float* ptr = hostGathered.data();
        int num_dets=0;
        for(int cl=0;cl<nc;cl++) num_dets+=keptIndices[b*nc+cl].size();
        dets[b]=detection_list_create((num_dets<8) ? 8 : num_dets);
        int num_person_detections=0;
        int num_face_detections=0;
        for(int cl=0;cl<nc;cl++)
        {
            int n=keptIndices[b*nc+cl].size();
            for(int i=0;i<n;i++)
            {
                detection_t *det=detection_list_add_end(dets[b]);
                assert(det!=0);
                float cx=ptr[0];
                float cy=ptr[1];
                float w=ptr[2];
                float h=ptr[3];
                det->x0=cx-w*0.5;
                det->x1=cx+w*0.5;
                det->y0=cy-h*0.5;
                det->y1=cy+h*0.5;
                det->index=keptIndices[b*nc+cl][i];
                det->cl=cl;
                det->conf=ptr[4+cl];
                det->num_face_points=0;
                det->num_pose_points=0;

                if (cl==inf->face_class_index)
                {
                    num_face_detections++;
                    det->num_face_points=5;
                    for(int k=0;k<det->num_face_points;k++)
                    {
                        det->face_points[k].x=ptr[4+nc+num_attributes+3*k+0];
                        det->face_points[k].y=ptr[4+nc+num_attributes+3*k+1];
                        det->face_points[k].conf=ptr[4+nc+num_attributes+3*k+2];
                    }
                }
                if (cl==inf->person_class_index)
                {
                    num_person_detections++;
                    det->num_pose_points=17;
                    for(int k=0;k<det->num_pose_points;k++)
                    {
                        det->pose_points[k].x=ptr[4+nc+num_attributes+3*(k+5)+0];
                        det->pose_points[k].y=ptr[4+nc+num_attributes+3*(k+5)+1];
                        det->pose_points[k].conf=ptr[4+nc+num_attributes+3*(k+5)+2];
                    }
                    det->num_attr=num_attributes;
                    for(int k=0;k<num_attributes;k++) det->attr[k]=ptr[4+nc+k];
                    if (inf->md.reid_vector_len!=0)
                    {
                        det->reid_vector_len=inf->md.reid_vector_len;
                        for(int i=0;i<inf->md.reid_vector_len;i++)
                        {
                            det->reid[i]=ptr[4+nc+num_attributes+3*(det->num_pose_points+det->num_face_points)+i];
                        }
                    }
                }
                ptr+=columns;
            }
        }
        assert(inf->person_class_index==0); // no issue if not, just too lazy to fix code below!
        assert(inf->face_class_index==1);

        dets[b]->person_dets=&dets[b]->det[0];
        dets[b]->num_person_detections=num_person_detections;
        dets[b]->face_dets=dets[b]->person_dets+num_person_detections;
        dets[b]->num_face_detections=num_face_detections;
    }
}

void infer_batch(infer_t *inf, image_t **img_list, detection_list_t **dets, int num)
{
    // check
    for(int i=0;i<num;i++)
    {
        assert(img_list[i]!=0);
    }
    // step 0: split into 'max batch' pieces
    int max_batch=std::min(inf->inf_limit_max_batch, inf->max_batch);
    if (num>max_batch)
    {
        for(int i=0;i<num;i+=max_batch) infer_batch(inf, img_list+i, dets+i, std::min(num-i, max_batch));
        return;
    }
    int inf_max_w=std::min(inf->inf_limit_max_width, inf->max_w);
    int inf_max_h=std::min(inf->inf_limit_max_height, inf->max_h);

    // step 1: determine overall inference size
    int max_w=0;
    int max_h=0;
    for(int i=0;i<num;i++)
    {
        image_t *img=img_list[i];
        int scale_w=0;
        int scale_h=0;

        determine_scale_size(img->width, img->height, inf_max_w, inf_max_h, &scale_w, &scale_h, 0,2,2,inf->inf_allow_upscale);

        max_w=std::max(max_w, scale_w);
        max_h=std::max(max_h, scale_h);
        debugf("%d) %dx%d->%dx%d\n",i,img->width,img->height,scale_w,scale_h);
    }
    inf_max_w=std::max(inf->inf_limit_min_width, inf_max_w);
    inf_max_h=std::max(inf->inf_limit_min_height, inf_max_h);
    int infer_w=std::max(inf->min_w, std::min(inf_max_w, (max_w+16)&(~31)));
    int infer_h=std::max(inf->min_h, std::min(inf_max_h, (max_h+16)&(~31)));
    debugf("INFER SIZE %dx%d (limit %dx%d)\n",infer_w,infer_h, inf->inf_limit_max_width, inf->inf_limit_max_height);

    bool do_testimage=false;

    image_t *testimage=0;
    if (do_testimage)
    {
        const char *testimage_path="/mldata/image/trt_inf.bin";
        log_debug("Loading inference test img %s",testimage_path);
        testimage=load_rgb32_tensor(testimage_path,640,640);
        display_image("test",testimage);
        infer_w=640;
        infer_h=640;
        usleep(1000000);
    }
    // step 2: determined scaled size for each image
    // the image will be fitted into the inference rect and padded on right
    // and bottom edges.
    int image_widths[num], image_heights[num];
    int offs_x[num], offs_y[num];
    image_t *image_scaled_conv[num];
    for(int i=0;i<num;i++)
    {
        image_t *img=(testimage!=0) ? testimage : img_list[i];
        determine_scale_size(img->width, img->height, infer_w, infer_h, image_widths+i,image_heights+i,
                          10, 2, 2, inf->inf_allow_upscale);
        //printf("%dx%d %dx%d %dx%d\n",infer_w,infer_h,img->width,img->height,image_widths[0],image_heights[0]);
        image_t *image_scaled=image_scale(img, image_widths[i], image_heights[i]);
        image_format_t fmt=image_scaled->format;
        if (  (fmt==IMAGE_FORMAT_RGB24_DEVICE)
            ||(fmt==IMAGE_FORMAT_RGB24_HOST)
            ||(fmt==IMAGE_FORMAT_RGB_PLANAR_FP32_HOST)
            ||(fmt==IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE))
            fmt=IMAGE_FORMAT_RGB24_DEVICE;
        else
            fmt=IMAGE_FORMAT_YUV420_DEVICE;
        image_scaled_conv[i]=image_convert(image_scaled, fmt);
        destroy_image(image_scaled);
    }

    if (testimage) destroy_image(testimage);

    // step 3: make planar RGB image with all batch images

    image_format_t fmt=inf->input_is_fp16 ? IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE : IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE;
    image_t *inf_image=image_make_tiled(fmt, infer_w, infer_h, image_scaled_conv, num, offs_x, offs_y);

    // step4: run inference
    auto input_dims = nvinfer1::Dims4{num, 3, infer_h, infer_w};
    assert(true==inf->ec->setInputShape(inf->input_tensor_name, input_dims));
    assert(true==inf->ec->setTensorAddress(inf->input_tensor_name, inf_image->rgb));
    int max_output_size=(int)inf->ec->getMaxOutputSize(inf->output_tensor_name);
    if (max_output_size>inf->output_size)
    {
        log_debug("Infer reallocate output memory [%d bytes, sz %dx%dx%d]\n",max_output_size,num,infer_h,infer_w);
        cuda_free(inf->output_mem);
        cuda_free_host(inf->output_mem_host);
        inf->output_mem=cuda_malloc(max_output_size);
        inf->output_mem_host=0;
        inf->output_size=max_output_size;
    }
    assert(true==inf->ec->setTensorAddress(inf->output_tensor_name, inf->output_mem));
    assert(true==inf->ec->allInputDimensionsSpecified());
    assert(0==inf->ec->inferShapes(0,0));

    Dims outputDims = inf->ec->getTensorShape(inf->output_tensor_name);
    assert(num==outputDims.d[0]); // batch
    int columns=outputDims.d[1]; // eg 110
    int rows=outputDims.d[2];   // eg 4620
    int numBoxes=rows;

    debugf("output dims %d %d %d\n",(int)outputDims.d[0],(int)outputDims.d[1],(int)outputDims.d[2]);

    cuda_stream_add_dependency(inf->stream, inf_image->stream);
    assert(true==inf->ec->enqueueV3(inf->stream));

    // step5: process detections
    if (inf->use_cuda_nms)
    {
        process_detections_cuda_nms(inf, num, columns, rows, dets);
    }
    else
    {
        if (inf->output_mem_host==0) inf->output_mem_host=cuda_malloc_host(inf->output_size);
        CHECK_CUDART_CALL(cudaMemcpyAsync(inf->output_mem_host, inf->output_mem, inf->output_size, cudaMemcpyDeviceToHost,inf->stream));
        CHECK_CUDART_CALL(cudaStreamSynchronize(inf->stream));
        float *p=(float *)inf->output_mem_host;
        for(int i=0;i<num;i++)
        {
            dets[i]=process_detections(inf,
                                    p+i*rows*columns, /*offset into batch*/
                                    rows,
                                    rows);
            debugf("i=%d (%dx%d, %d,%d,%d)\n",i,image_widths[i], image_heights[i],rows,columns,inf->output_size);
        }
    }

    // map detections to the original image

    for(int i=0;i<num;i++)
    {
        detection_list_scale_add2(dets[i], 1.0/image_widths[i],
                                       1.0/image_heights[i],
                                       offs_x[i],
                                       offs_y[i]);
    }

    for(int i=0;i<num;i++)
    {
        detection_list_generate_overlap_masks(dets[i]);
        if (inf->do_fuse_face_person) detection_list_fuse_face_person(dets[i]);
    }

    destroy_image(inf_image);
    for(int i=0;i<num;i++) destroy_image(image_scaled_conv[i]);
}

detection_list_t *infer(infer_t *inf, image_t *img)
{
    detection_list_t *ret[1];
    image_t *images[1];
    images[0]=img;
    infer_batch(inf, images, ret, 1);
    return ret[0];
}

void infer_configure(infer_t *inf, infer_config_t *config)
{
    if (!inf) return;
    if (!config) return;
    if (config->set_det_thr) inf->det_thr=config->det_thr;
    if (config->set_nms_thr) inf->nms_thr=config->nms_thr;
    if (config->set_use_cuda_nms) inf->use_cuda_nms=config->use_cuda_nms;
    if (config->set_limit_max_batch) inf->inf_limit_max_batch=config->limit_max_batch;
    if (config->set_limit_min_width) inf->inf_limit_min_width=config->limit_min_width;
    if (config->set_limit_min_height) inf->inf_limit_min_height=config->limit_min_height;
    if (config->set_limit_max_width) inf->inf_limit_max_width=config->limit_max_width;
    if (config->set_limit_max_height) inf->inf_limit_max_height=config->limit_max_height;
    if (config->set_fuse_face_person) inf->do_fuse_face_person=config->fuse_face_person;
    if (config->set_allow_upscale) inf->inf_allow_upscale=config->allow_upscale;
    if (config->set_max_detections)
    {
        if (inf->nms) cuda_nms_free_workspace(inf->nms);
        inf->nms_max_boxes=0;
        inf->nms=0;
        inf->max_detections=config->max_detections;
    }
}
