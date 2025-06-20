
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include "cuviddec.h"
#include "nvcuvid.h"
#include "simple_decoder.h"
#include "cuda_stuff.h"
#include "log.h"

#if (UBONCSTUFF_PLATFORM == 0) // Desktop Nvidia GPU
struct simple_decoder
{
    int coded_width;
    int coded_height;
    int out_width;
    int out_height;
    int target_width;
    int target_height;
    void *context;
    uint64_t time;
    uint64_t time_increment;
    //CUstream stream;
    CUvideodecoder decoder;
    CUvideoparser videoParser;
    CUVIDEOFORMAT videoFormat;
    void (*frame_callback)(void *context, image_t *decoded_frame);
};

int CUDAAPI HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pFormat)
{
    simple_decoder_t *dec = (simple_decoder_t *)pUserData;

    if (dec->decoder != NULL && (pFormat->coded_width != dec->coded_width || pFormat->coded_height != dec->coded_height))
    {
        // Resolution change, reinitialize the decoder
        CHECK_CUDA_CALL(cuvidDestroyDecoder(dec->decoder));
        dec->decoder = NULL;
    }

    if (dec->decoder == NULL)
    {
        // Initialize decoder with the new format
        dec->videoFormat = *pFormat;
        dec->coded_width = pFormat->coded_width;
        dec->coded_height = pFormat->coded_height;

        if (dec->target_width==0)
        {
            dec->out_width=pFormat->display_area.right-pFormat->display_area.left;
            dec->out_height=pFormat->display_area.bottom-pFormat->display_area.top;
        }
        else
        {
            dec->out_width=dec->target_width;
            dec->out_height=dec->target_height;
        }

        log_debug("Create cuda decoder %dx%d; display area (%d,%d)-(%d,%d) output %dx%d codec %d\n",
            dec->coded_width,dec->coded_height,
            pFormat->display_area.left,pFormat->display_area.top,
            pFormat->display_area.right,pFormat->display_area.bottom,
            dec->out_width,dec->out_height, (int)pFormat->codec);

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
        decodeCreateInfo.display_area.left=pFormat->display_area.left;
        decodeCreateInfo.display_area.top=pFormat->display_area.top;
        decodeCreateInfo.display_area.right=pFormat->display_area.right;
        decodeCreateInfo.display_area.bottom=pFormat->display_area.bottom;
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

void simple_decoder_set_framerate(simple_decoder_t *dec, double fps)
{
    dec->time_increment=(uint64_t)(90000.0/fps);
}

int CUDAAPI HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo)
{
    simple_decoder_t *dec=(simple_decoder_t *)pUserData;
    image_t *dec_img=create_image_no_surface_memory(dec->out_width, dec->out_height, IMAGE_FORMAT_NV12_DEVICE);
    CUdeviceptr decodedFrame=0;
    CUVIDPROCPARAMS videoProcessingParameters = {};
    videoProcessingParameters.progressive_frame = pDispInfo->progressive_frame;
    videoProcessingParameters.second_field =  (pDispInfo->repeat_first_field != 0);;
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
    img->timestamp=dec->time;
    dec->time+=dec->time_increment;
    dec->frame_callback(dec->context, img);
    destroy_image(img);
    destroy_image(dec_img);

    return 1;
}

simple_decoder_t *simple_decoder_create(void *context,
                                        void (*frame_callback)(void *context, image_t *decoded_frame),
                                        simple_decoder_codec_t codec)
{
    check_cuda_inited();
    simple_decoder_t *dec = (simple_decoder_t *)malloc(sizeof(simple_decoder_t));
    if (dec==0) return 0;
    memset(dec, 0, sizeof(simple_decoder_t));
    dec->frame_callback=frame_callback;
    dec->context = context;
    dec->target_width=0;
    dec->target_height=0;
    dec->out_width=1280;
    dec->out_height=720;
    dec->time_increment=90000/30;
    dec->time=0;

    CUVIDPARSERPARAMS videoParserParams;
    memset(&videoParserParams,0,sizeof(videoParserParams));
    videoParserParams.CodecType = (codec==SIMPLE_DECODER_CODEC_H264) ? cudaVideoCodec_H264 : cudaVideoCodec_HEVC;
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
        if (dec->videoParser)
        {
            CHECK_CUDA_CALL(cuvidDestroyVideoParser(dec->videoParser));
        }
        if (dec->decoder)
        {
            CHECK_CUDA_CALL(cuvidDestroyDecoder(dec->decoder));
        }
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
