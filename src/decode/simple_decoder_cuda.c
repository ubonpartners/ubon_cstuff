
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include "nvdecode/cuviddec.h"
#include "nvdecode/nvcuvid.h"
#include "simple_decoder.h"
#include "cuda_stuff.h"
#include "log.h"
#include "misc.h"
#include "yaml_stuff.h"
#include "profile.h"

#define debugf if (0) log_error

#define MAX_DECODE_W    3840
#define MAX_DECODE_H    2160

#if (UBONCSTUFF_PLATFORM == 0) // Desktop Nvidia GPU
struct simple_decoder
{
    int coded_width, coded_height;
    int out_width, out_height;
    int target_width, target_height;
    int scaled_width, scaled_height;
    simple_decoder_codec_t codec;
    void *context;
    image_format_t output_format;
    double time;
    bool destroyed;
    bool use_frame_times;
    bool low_latency;
    bool force_skip;
    //CUstream stream;
    CUvideoctxlock vidlock;
    CUvideodecoder decoder;
    CUvideoparser videoParser;
    CUVIDEOFORMAT videoFormat;
    void (*frame_callback)(void *context, image_t *decoded_frame);

    int constraint_max_width;
    int constraint_max_height;
    double constraint_min_time_delta;
    double last_output_time;

    uint64_t stats_bytes_decoded;
    uint64_t stats_macroblocks_decoded;
    uint32_t stats_frames_decoded;
    uint32_t stats_frames_output_skipped;
    uint32_t stats_output_time_reset;
    uint32_t stats_unconcealable_decode_errors;
    uint32_t stats_concealable_decode_errors;
    uint32_t stats_resolution_changes;
};

int CUDAAPI HandleVideoSequence(void *pUserData, CUVIDEOFORMAT *pFormat)
{
    simple_decoder_t *dec = (simple_decoder_t *)pUserData;

    if (dec->decoder != NULL && (pFormat->coded_width != dec->coded_width || pFormat->coded_height != dec->coded_height))
    {
        // Resolution change, reinitialize the decoder
        dec->stats_resolution_changes++;
        CHECK_CUDA_CALL(cuvidDestroyDecoder(dec->decoder));
        dec->decoder = NULL;
    }

    if (dec->decoder == NULL)
    {
        // Initialize decoder with the new format
        dec->videoFormat = *pFormat;
        dec->coded_width = pFormat->coded_width;
        dec->coded_height = pFormat->coded_height;

        if ((pFormat->coded_width>MAX_DECODE_W)||(pFormat->coded_height>MAX_DECODE_H))
        {
            log_error("cuda decode cannot decode (too big) %dx%d",pFormat->coded_width,pFormat->coded_height);
            return 0;
        }

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

        debugf("Create cuda decoder %dx%d; display area (%d,%d)-(%d,%d) output %dx%d codec %d",
            dec->coded_width,dec->coded_height,
            pFormat->display_area.left,pFormat->display_area.top,
            pFormat->display_area.right,pFormat->display_area.bottom,
            dec->out_width,dec->out_height, (int)pFormat->codec);

        CUVIDDECODECREATEINFO decodeCreateInfo = {0};
        decodeCreateInfo.CodecType = pFormat->codec;
        decodeCreateInfo.ulWidth = pFormat->coded_width;
        decodeCreateInfo.ulHeight = pFormat->coded_height;
        decodeCreateInfo.ulTargetWidth = dec->out_width;
        decodeCreateInfo.ulTargetHeight = dec->out_height;
        decodeCreateInfo.ulMaxWidth = pFormat->coded_width;
        decodeCreateInfo.ulMaxHeight = pFormat->coded_height;
        decodeCreateInfo.ulNumDecodeSurfaces = pFormat->min_num_decode_surfaces;
        decodeCreateInfo.ulNumOutputSurfaces = 1;
        decodeCreateInfo.display_area.left=pFormat->display_area.left;
        decodeCreateInfo.display_area.top=pFormat->display_area.top;
        decodeCreateInfo.display_area.right=pFormat->display_area.right;
        decodeCreateInfo.display_area.bottom=pFormat->display_area.bottom;
        decodeCreateInfo.ChromaFormat = pFormat->chroma_format;
        decodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
        decodeCreateInfo.bitDepthMinus8 = pFormat->bit_depth_luma_minus8;
        decodeCreateInfo.vidLock = dec->vidlock;
        decodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
        decodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;

        CHECK_CUDA_CALL(cuvidCreateDecoder(&dec->decoder, &decodeCreateInfo));
    }
    return ((dec->low_latency && dec->codec==SIMPLE_DECODER_CODEC_H264)) ? pFormat->min_num_decode_surfaces : pFormat->min_num_decode_surfaces; // override the parser DPB size; who knew!
}

static int CUDAAPI HandlePictureDecode(void *pUserData, CUVIDPICPARAMS *pPicParams)
{
    simple_decoder_t *dec=(simple_decoder_t *)pUserData;
    debugf("Handle picture decode");
    CHECK_CUDA_CALL(cuvidDecodePicture(dec->decoder, pPicParams));
    return 1;
}

int CUDAAPI HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo)
{
    debugf("Handle picture display");
    simple_decoder_t *dec=(simple_decoder_t *)pUserData;
    if (dec->destroyed) return 1; // ignore if "destroy" already called

    dec->stats_frames_decoded++;
    dec->stats_macroblocks_decoded+=((dec->out_width+15)>>4)*((dec->out_height+15)>>4);

    CUdeviceptr decodedFrame=0;
    CUVIDPROCPARAMS videoProcessingParameters = {0};
    videoProcessingParameters.progressive_frame = pDispInfo->progressive_frame;
    videoProcessingParameters.second_field =  (pDispInfo->repeat_first_field != 0);
    videoProcessingParameters.top_field_first = pDispInfo->top_field_first;
    videoProcessingParameters.unpaired_field = pDispInfo->repeat_first_field < 0;
    unsigned int pitch;

    double time=pDispInfo->timestamp/90000.0;

    bool skip=dec->force_skip;
    dec->force_skip=false;

    if (dec->constraint_min_time_delta!=0)
    {
        float delta=time-dec->last_output_time;
        if ((delta>10.0)||(delta<0))
        {
            if (dec->stats_frames_decoded>1) log_error("decoder time constraint unexpected delta %f->%f; resetting",dec->last_output_time,time);
            dec->last_output_time=time;
            dec->stats_output_time_reset++;
        }
        else
        {
            skip|=(delta<dec->constraint_min_time_delta);
        }
    }
    if (!skip) dec->last_output_time=time;

    if (skip)
    {
        debugf("decoder skip last %f this %f delta %f min %f",dec->last_output_time,time,time-dec->last_output_time,dec->constraint_min_time_delta);
        // early return if the frame not needed, avoid a lot of work
        dec->stats_frames_output_skipped++;
        return 1;
    }

    image_t *dec_img=(IMAGE_FORMAT_NV12_DEVICE!=dec->output_format) ? image_create_no_surface_memory(dec->out_width, dec->out_height, IMAGE_FORMAT_NV12_DEVICE)
                                                                    : image_create(dec->out_width, dec->out_height, IMAGE_FORMAT_NV12_DEVICE);
    image_t *out_img=0;
    videoProcessingParameters.output_stream = dec_img->stream;
    CHECK_CUDA_CALL(cuvidMapVideoFrame(dec->decoder, pDispInfo->picture_index, &decodedFrame, &pitch, &videoProcessingParameters));

    CUVIDGETDECODESTATUS decodeStatus;
    CHECK_CUDA_CALL(cuvidGetDecodeStatus(dec->decoder, pDispInfo->picture_index, &decodeStatus));
    if ((decodeStatus.decodeStatus!=cuvidDecodeStatus_Success)
        && (decodeStatus.decodeStatus!=cuvidDecodeStatus_Error_Concealed))
    {
        log_error("Cuda decoder error %d",(int)decodeStatus.decodeStatus);
        // un-concealable decoder error
        dec->stats_unconcealable_decode_errors++;
        CHECK_CUDA_CALL(cuvidUnmapVideoFrame(dec->decoder, decodedFrame));
        return 1;
    }

    if (decodeStatus.decodeStatus==cuvidDecodeStatus_Error_Concealed)
    {
        log_info("Cuda decoder error concealment");
        dec->stats_concealable_decode_errors++;
    }

    // cuda decoder outputs NV12. If we are also asking for NV12 then we need to copy it.
    // Sadly, the copy is needed as the data only stays valid inside the map video
    // for other output formats we can wrap the cuda output in an image structure and call convert
    // the convert effectively 'copies' the data

    if (IMAGE_FORMAT_NV12_DEVICE==dec->output_format) {
        CHECK_CUDART_CALL(cudaMemcpy2D(
            dec_img->y, dec_img->stride_y,
            (void*)decodedFrame, pitch,
            dec_img->width, (dec_img->height*3)/2,
            cudaMemcpyDeviceToDevice
        ));
        out_img=image_reference(dec_img);
    }
    else {
        dec_img->y=(uint8_t*)decodedFrame;
        dec_img->u=(uint8_t*)decodedFrame+pitch*dec->out_height;
        dec_img->v=dec_img->u+1;
        dec_img->stride_y=dec_img->stride_uv=pitch;
        out_img=image_convert(dec_img, dec->output_format);
        image_sync(out_img);
    }
    out_img->meta.time=time;
    out_img->meta.capture_realtime=profile_time();
    out_img->meta.flags=MD_CAPTURE_REALTIME_SET;

    CHECK_CUDA_CALL(cuvidUnmapVideoFrame(dec->decoder, decodedFrame));
    if (file_trace_enabled)
    {
        FILE_TRACE("Simple decoder %dx%d fmt %d TS %f hash %lx",out_img->width,out_img->height,out_img->format,out_img->meta.time,(out_img==0) ? 0 : image_hash(out_img));
    }

    image_t *scaled_out_img=0;
    if (dec->constraint_max_width!=0 && dec->constraint_max_height!=0) {
        determine_scale_size(out_img->width, out_img->height,
                            dec->constraint_max_width, dec->constraint_max_width,
                            &dec->scaled_width, &dec->scaled_height,
                            10, 8, 8, false);
        scaled_out_img=image_scale(out_img, dec->scaled_width, dec->scaled_height);
    }
    else {
        dec->scaled_width=out_img->width;
        dec->scaled_height=out_img->height;
        scaled_out_img=image_reference(out_img);
    }
    debugf("decoder output t=%f\n",scaled_out_img->meta.time);
    dec->frame_callback(dec->context, scaled_out_img);
    if (scaled_out_img) image_destroy(scaled_out_img);
    if (out_img) image_destroy(out_img);
    if (dec_img) image_destroy(dec_img);
    return 1;
}

simple_decoder_t *simple_decoder_create(void *context,
                                        void (*frame_callback)(void *context, image_t *decoded_frame),
                                        simple_decoder_codec_t codec,
                                        bool low_latency)
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
    dec->low_latency=low_latency;
    dec->output_format=IMAGE_FORMAT_YUV420_DEVICE;
    dec->codec=codec;
    dec->last_output_time=-5.0;

    CHECK_CUDA_CALL(cuvidCtxLockCreate(&dec->vidlock, get_CUcontext()));

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

void simple_decoder_set_output_format(simple_decoder_t *dec, image_format_t fmt)
{
    dec->output_format=fmt;
}

void simple_decoder_destroy(simple_decoder_t *dec)
{
    if (dec)
    {
        if (dec->decoder)
        {
            dec->destroyed=true;
            CHECK_CUDA_CALL(cuvidDestroyDecoder(dec->decoder));
        }
        CHECK_CUDA_CALL(cuvidCtxLockDestroy(dec->vidlock));
        free(dec);
    }
}

void simple_decoder_decode(simple_decoder_t *dec, uint8_t *bitstream_data, int data_size, double frame_time, bool force_skip)
{
    debugf("decode %d bytes; time=%f",data_size,frame_time);
    CUVIDSOURCEDATAPACKET packet = {0};
    packet.payload=bitstream_data;
    packet.payload_size=data_size;
    if (frame_time>=0)
    {
        packet.timestamp=frame_time*90000;
        packet.flags=CUVID_PKT_TIMESTAMP;
        dec->use_frame_times=true;
    }
    else
    {
        dec->use_frame_times=false;
    }
    dec->stats_bytes_decoded+=data_size;
    if (force_skip) dec->force_skip=true;
    CHECK_CUDA_CALL(cuvidParseVideoData(dec->videoParser, &packet));
}

void simple_decoder_constrain_output(simple_decoder_t *dec, int max_width, int max_height, double min_time_delta)
{
    dec->constraint_max_width=max_width;
    dec->constraint_max_height=max_height;
    dec->constraint_min_time_delta=min_time_delta;
}

YAML::Node simple_decoder_get_stats(simple_decoder *dec)
{
    YAML::Node root;
    root["codec"]=(dec->codec==SIMPLE_DECODER_CODEC_H264) ? "H264" : "H265";
    root["current_coded_width"]=dec->coded_width;
    root["current_coded_height"]=dec->coded_height;
    root["current_decoder_output_width"]=dec->out_width;
    root["current_decoder_output_height"]=dec->out_height;
    root["current_scaled_output_width"]=dec->scaled_width;
    root["current_scaled_output_height"]=dec->scaled_height;
    root["bytes_decoded"]=dec->stats_bytes_decoded;
    root["macroblocks_decoded"]=dec->stats_macroblocks_decoded;
    root["frames_decoded"]=dec->stats_frames_decoded;
    root["frames_output_skipped"]=dec->stats_frames_output_skipped;
    root["output_time_reset"]=dec->stats_output_time_reset;
    root["unconcealable_decode_errors"]=dec->stats_unconcealable_decode_errors;
    root["concealable_decode_errors"]=dec->stats_concealable_decode_errors;
    root["resolution_changes"]=dec->stats_resolution_changes;
    return root;
}

#endif //(UBONCSTUFF_PLATFORM == 0)
