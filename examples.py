import numpy as np
import ubon_pycstuff.ubon_pycstuff as upyc
import cv2
import stuff

def basic_test(jpeg_file):
    bgr_img = cv2.imread("/mldata/image/arrest.jpg")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # copy image to C domain
    img = upyc.c_image.from_numpy(rgb_img) # will be RGB24_HOST

    #scale to 720p
    img_scaled = img.scale(1280, 720) # implicily will get copied to GPU and converted to YUV420

    # copy back and display
    round_trip_bgr = cv2.cvtColor(img_scaled.to_numpy(), cv2.COLOR_RGB2BGR)

    cv2.imshow("Round-trip image", round_trip_bgr)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def test_inference(jpeg_file):
    bgr_img = cv2.imread("/mldata/image/arrest.jpg")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = upyc.c_image.from_numpy(rgb_img) # will be RGB24_HOST

    # run tensortRT inference
    inf = upyc.c_infer("/mldata/weights/trt/yolo11l-dpa-131224.trt", "")
    dets=inf.run(img)
    for d in dets:
        print(d)

    display=stuff.Display(1280,720)
    for d in dets:
        display.draw_box(d["box"], thickness=2, clr=(255, 255, 0, 0))
    display.show(rgb_img, is_rgb=True)
    display.get_events(5000)

def test_nvof(h264_file):
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

def test_motiondet(h264_file):
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

basic_test("/mldata/image/arrest.jpg")
test_inference("/mldata/image/arrest.jpg")
test_nvof("/mldata/video/MOT20-01.264")
test_motiondet("/mldata/tracking/cevo_april25/video/generated_h264/INof_LD_Out_Light_FFcam_002.h264")


