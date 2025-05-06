
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include "cuviddec.h"
#include "nvcuvid.h"
#include "simple_decoder.h"
#include "cuda_stuff.h"

#if (UBONCSTUFF_PLATFORM == 0) // Desktop Nvidia GPU
struct simple_decoder 
{
    int width;
    int height;
    int out_width;
    int out_height;
    void *context;
    //CUstream stream;
    CUvideodecoder decoder;
    CUvideoparser videoParser;
    CUVIDEOFORMAT videoFormat;
    void (*frame_callback)(void *context, image_t *decoded_frame);
};

int CUDAAPI HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pFormat) 
{
    simple_decoder_t *dec = (simple_decoder_t *)pUserData;

    if (dec->decoder != NULL && (pFormat->coded_width != dec->width || pFormat->coded_height != dec->height)) 
    {
        // Resolution change, reinitialize the decoder
        CHECK_CUDA_CALL(cuvidDestroyDecoder(dec->decoder));
        dec->decoder = NULL;
    }

    if (dec->decoder == NULL) 
    {
        // Initialize decoder with the new format
        dec->videoFormat = *pFormat;
        dec->width = pFormat->coded_width;
        dec->height = pFormat->coded_height;
        printf("Create cuda decoder %dx%d\n",dec->width,dec->height);

        CUVIDDECODECREATEINFO decodeCreateInfo = {0};
        decodeCreateInfo.CodecType = pFormat->codec;
        decodeCreateInfo.ulWidth = pFormat->coded_width;
        decodeCreateInfo.ulHeight = pFormat->coded_height;
        decodeCreateInfo.ulTargetWidth = dec->out_width;//pFormat->coded_width;
        decodeCreateInfo.ulTargetHeight = dec->out_height;//pFormat->coded_height;
        decodeCreateInfo.ulMaxWidth = 1920;
        decodeCreateInfo.ulMaxHeight = 1088;
        decodeCreateInfo.ulNumDecodeSurfaces = 8;
        decodeCreateInfo.ulNumOutputSurfaces = 2;
        decodeCreateInfo.ChromaFormat = pFormat->chroma_format;
        decodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
        decodeCreateInfo.bitDepthMinus8 = pFormat->bit_depth_luma_minus8;
        decodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
        decodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;

        CHECK_CUDA_CALL(cuvidCreateDecoder(&dec->decoder, &decodeCreateInfo));
    }
    return 1;
}

static int CUDAAPI HandlePictureDecode(void *pUserData, CUVIDPICPARAMS *pPicParams)
{
    simple_decoder_t *dec=(simple_decoder_t *)pUserData;
    CHECK_CUDA_CALL(cuvidDecodePicture(dec->decoder, pPicParams));
    return 1;
}

int CUDAAPI HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo) 
{
    simple_decoder_t *dec=(simple_decoder_t *)pUserData;
    image_t *dec_img=create_image_no_surface_memory(dec->out_width, dec->out_height, IMAGE_FORMAT_NV12_DEVICE);
    CUdeviceptr decodedFrame=0;
    CUVIDPROCPARAMS videoProcessingParameters = {};
    videoProcessingParameters.progressive_frame = pDispInfo->progressive_frame;
    videoProcessingParameters.second_field = pDispInfo->repeat_first_field + 1;
    videoProcessingParameters.top_field_first = pDispInfo->top_field_first;
    videoProcessingParameters.unpaired_field = pDispInfo->repeat_first_field < 0;
    videoProcessingParameters.output_stream = dec_img->stream;
    unsigned int pitch;
    CHECK_CUDA_CALL(cuvidMapVideoFrame(dec->decoder, pDispInfo->picture_index, &decodedFrame, &pitch, &videoProcessingParameters));
    dec_img->y=(uint8_t*)decodedFrame;
    dec_img->u=(uint8_t*)decodedFrame+pitch*dec->out_height;
    dec_img->v=dec_img->u+1;
    dec_img->stride_y=dec_img->stride_uv=pitch;
    image_t *img=image_convert(dec_img, IMAGE_FORMAT_YUV420_DEVICE);
    CHECK_CUDA_CALL(cuvidUnmapVideoFrame(dec->decoder, decodedFrame));
    dec->frame_callback(dec->context, img);
    destroy_image(img);

    return 1;
}

simple_decoder_t *simple_decoder_create(void *context, void (*frame_callback)(void *context, image_t *decoded_frame)) 
{
    check_cuda_inited();
    simple_decoder_t *dec = (simple_decoder_t *)malloc(sizeof(simple_decoder_t));
    if (dec==0) return 0;
    memset(dec, 0, sizeof(simple_decoder_t));
    dec->frame_callback=frame_callback;
    dec->context = context;
    dec->out_width=1280;
    dec->out_height=720;
    //CHECK_CUDA_CALL(cuStreamCreate(&dec->stream, CU_STREAM_DEFAULT));
    CUVIDPARSERPARAMS videoParserParams;
    memset(&videoParserParams,0,sizeof(videoParserParams));
    videoParserParams.CodecType = cudaVideoCodec_H264; // Change to cudaVideoCodec_HEVC for H.265
    videoParserParams.ulMaxNumDecodeSurfaces = 8;//
    videoParserParams.ulClockRate = 1000;
    videoParserParams.ulErrorThreshold = 100;
    videoParserParams.ulMaxDisplayDelay = 0;
    videoParserParams.pUserData = dec;
    videoParserParams.pfnSequenceCallback = HandleVideoSequence;
    videoParserParams.pfnDecodePicture = HandlePictureDecode;
    videoParserParams.pfnDisplayPicture = HandlePictureDisplay;
    CHECK_CUDA_CALL(cuvidCreateVideoParser(&dec->videoParser, &videoParserParams));
    return dec;
}

void simple_decoder_destroy(simple_decoder_t *dec) 
{
    if (dec) 
    {
        //CHECK_CUDA_CALL(cuStreamDestroy(dec->stream));
        CHECK_CUDA_CALL(cuvidDestroyVideoParser(dec->videoParser));
        CHECK_CUDA_CALL(cuvidDestroyDecoder(dec->decoder));
        free(dec);
    }
}

void simple_decoder_decode(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size)
{
    CUVIDSOURCEDATAPACKET packet = {0};
    packet.payload=bitstream_data;
    packet.payload_size=data_size;
    CHECK_CUDA_CALL(cuvidParseVideoData(dec->videoParser, &packet));
}
#endif //(UBONCSTUFF_PLATFORM == 0)

