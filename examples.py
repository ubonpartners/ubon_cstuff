import numpy as np
import ubon_pycstuff.ubon_pycstuff as upyc
from PIL import Image
import stuff
import time
import os
import sys
import argparse

def argument_parser():
    parser = argparse.ArgumentParser("ubon_cstuff examples")
    parser.add_argument("--basic_test", dest="basic_test",
        help="Basic Test <jpg file>", default=None, type=str)
    parser.add_argument("--basic_test_mono", dest="basic_test_mono",
        help="Basic Test Mono <jpg file>", default=None, type=str)
    parser.add_argument("--test_inference", dest="test_inference",
        help="Test Inference <jpg file>", default=None, type=str)
    parser.add_argument("--test_nvof", dest="test_nvof",
        help="Test NVOF <h264 nal file>", default=None, type=str)
    parser.add_argument("--test_motiondet", dest="test_motiondet",
        help="Test Motion Detection <h264 nal file>", default=None, type=str)
    parser.add_argument("--trt_model", dest="trt_model",
        help="Tensor RT Model to use <model.trt file>", default=None, type=str)
    parser.print_help()
    args = parser.parse_args()
    version = upyc.get_version()
    print(f"args = {args}\n")
    print(f"version = {version}\n")
    # validate all given files do exist.
    exist = True
    for k, v in vars(args).items():
        if v is not None and not os.path.exists(v):
            print(f"argument --{k} {v} does not exist")
            exist = False
    if exist is False:
        print("Exiting since one of the input file is not existing")
        sys.exit(2)
    return args

def basic_test(jpeg_file="/mldata/image/arrest.jpg"):
    rgb_img = np.asarray(Image.open(jpeg_file).convert("RGB"))

    # copy image to C domain
    img = upyc.c_image.from_numpy(rgb_img) # will be RGB24_HOST
    print(img)
    #scale to 720p
    img_scaled = img.scale(1280, 720) # implicily will get copied to GPU and converted to YUV420

    # copy back and display
    Image.fromarray(img_scaled.to_numpy()).show()
    time.sleep(5)

def basic_test_mono(jpeg_file="/mldata/image/arrest.jpg"):
    rgb_img = np.asarray(Image.open(jpeg_file).convert("RGB"))
    # copy image to C domain
    img = upyc.c_image.from_numpy(rgb_img)
    img=img.convert(upyc.YUV420_DEVICE)
    print(img)
    img.sync()
    #img=img.convert(upyc.MONO_DEVICE)
    #img.sync()
    #img=img.convert(upyc.YUV420_DEVICE)
    #img.sync()
    #img.display("Mono image")
    img=img.scale(320,256)
    img.display("Mono image scaled")
    time.sleep(10)

def test_inference(jpeg_file="/mldata/image/arrest.jpg", trt_model="/mldata/weights/trt/yolo11l-dpa-131224.trt"):
    rgb_img = np.asarray(Image.open(jpeg_file).convert("RGB"))
    img = upyc.c_image.from_numpy(rgb_img) # will be RGB24_HOST
    print(img)

    # run tensortRT inference
    inf = upyc.c_infer(trt_model, "")
    dets=inf.run(img)
    for d in dets:
        print(d)

    display=stuff.Display(1280,720)
    for d in dets:
        display.draw_box(d["box"], thickness=2, clr=(255, 255, 0, 0))
    display.show(rgb_img, is_rgb=True)
    display.get_events(5000)

def test_nvof(h264_file="/mldata/video/MOT20-01.264"):
    # create a NVDEC video decoder
    decoder = upyc.c_decoder()
    with open(h264_file, "rb") as f:
        bitstream = f.read()

    # decode some video
    frames = decoder.decode(bitstream)

    # create an NVOF optical flow engine
    flow_engine = upyc.c_nvof(320, 320)

    d=stuff.Display(1280,720)

    for i, frame in enumerate(frames):
        if i%10!=0:
            continue
        print(f"frame {i}")
        arr = frame.to_numpy()

        costs, flow = flow_engine.run(frame) # returns np arrays with costs, flow vectors
        h,w=costs.shape
        d.clear()

        for y in range(h):
            for x in range(w):
                box=[x/w, y/h, (x+1)/w, (y+1)/h]
                v=int(costs[y][x])*5
                clr=(v, 255, 255, 255)
                d.draw_box(box, thickness=-1, clr=(v, 255, 0, 0))
                start=[(x+0.5)/w, (y+0.5)/h]
                dx=flow[y][x][0]
                dy=flow[y][x][1]

                stop=[start[0]+flow[y][x][0], start[1]+flow[y][x][1]]
                if abs(dx)>0.001 or abs(dy)>0.001:
                    d.draw_line(start, stop, clr=(255,255,255,255))
        d.show(arr, is_rgb=True)
        d.get_events(30)

def test_motiondet(h264_file="/mldata/tracking/cevo_april25/video/generated_h264/INof_LD_Out_Light_FFcam_002.h264"):
    # create a NVDEC video decoder
    # blur the frames and take frame differences
    # display the differences

    decoder = upyc.c_decoder()
    with open(h264_file, "rb") as f:
        bitstream = f.read()

    # decode some video
    frames = decoder.decode(bitstream)

    d=stuff.Display(1280,720)
    last=None
    for i, frame in enumerate(frames):
        if i%5!=0:
            continue
        blurred=frame.blur()
        if last is not None:
            mad=blurred.mad_4x4(last)
        print(f"frame {i}")
        if last is not None:
            arr = mad.to_numpy()
            d.show(arr, is_rgb=True)
            d.get_events(30)
        last=blurred

def test_track(h264_file="/mldata/tracking/cevo_april25/video/generated_h264/INof_LD_Out_Light_FFcam_002.h264"):

    decoder = upyc.c_decoder(upyc.SIMPLE_DECODER_CODEC_H264)
    with open(h264_file, "rb") as f:
        bitstream = f.read()

    # decode some video
    frames = decoder.decode(bitstream)

    display=stuff.Display(1280,720)
    track_shared=upyc.c_track_shared_state("/mldata/config/track/trackers/uc_reid.yaml")
    track_stream=upyc.c_track_stream(track_shared)
    track_stream.run_on_images(frames)
    track_results=track_stream.get_results()
    for i,r in enumerate(track_results):
        img=frames[i].to_numpy()
        if 'track_dets' in r and r['track_dets'] is not None:
            dets=r['track_dets']
            display.clear()
            for d in dets:
                display.draw_box(d["box"], thickness=2, clr=(255, 255, 0, 0))
        display.show(img, is_rgb=True)
        display.get_events(10)

args = argument_parser()

upyc.cuda_set_sync_mode(False, False)
if args.basic_test is not None:
    print(f"basic_test: {args.basic_test}")
    basic_test(args.basic_test)
if args.basic_test_mono is not None:
    print(f"basic_test_mono: {args.basic_test_mono}")
    basic_test_mono(args.basic_test_mono)
if args.test_inference is not None and args.trt_model is not None:
    print(f"test_inference: {args.test_inference}")
    test_inference(args.test_inference, args.trt_model)
if args.test_nvof is not None:
    print(f"test_nvof: {args.test_nvof}")
    test_nvof(args.test_nvof)
if args.test_motiondet is not None:
    print(f"test_motiondet: {args.test_motiondet}")
    test_motiondet(args.test_motiondet)

print("completed")
