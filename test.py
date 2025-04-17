import numpy as np
import ubon_pycstuff.ubon_pycstuff as upyc
import cv2
import stuff


def test_assert(val, txt):
    ret="PASS"
    if val==False:
        ret="FAIL"
    print(f"{txt:50s} : {ret}")

def test_hash(numpy_img):
    # basic test that we can has a video surface and we get the same hash
    # value for the surface in GPU and host memory
    upyc.cuda_set_sync_mode(False, False)
    img = upyc.c_image.from_numpy(numpy_img).convert(upyc.RGB24_HOST)
    img_device=img.convert(upyc.RGB24_DEVICE)
    img_host2=img_device.convert(upyc.RGB24_HOST)
    test_assert(img.hash()==img_device.hash(), "host mem hash==device mem hash")
    test_assert(img_host2.hash()==img_device.hash(), "host mem hash2==host mem hash")

def test_reproducibility(numpy_img):
    # tests for reproducibility - e.g. this will usually find cuda syncronization
    # errors. We do various chains of surface operations and check the resulting 
    # hash is the same when we do the same set of ops from the same starting surface
    # note, because of padding any surfaces should be a multiple of 32 pixels wide
    # else the hashes may disagree

    upyc.cuda_set_sync_mode(False, False)
    img = upyc.c_image.from_numpy(numpy_img)

    for run in range(10):
        hashes=[]
        for i in range(20):
            test=img
            if run==0:
                test=test.convert(upyc.YUV420_DEVICE)
            if run>=1:
                test=test.scale(1280, 720)
            if (run>=2):
                test=test.scale(320,256)
            if run==9:
                test=test.scale(1280,720)
            if run==3 or run==4 or run==5:
              test=test.convert(upyc.NV12_DEVICE)
            if run==4 or run==5:
                test=test.convert(upyc.YUV420_DEVICE)
            if run==5:
                test=test.convert(upyc.YUV420_HOST)
            if run==6:
                test=test.convert(upyc.RGB24_HOST)
            if run==7:
                test=test.convert(upyc.RGB_PLANAR_FP32_DEVICE)
            if run==8:
                test=test.convert(upyc.RGB_PLANAR_FP32_DEVICE)
            hashes.append(test.hash())

        test_assert(all(x == hashes[0] for x in hashes), f"reproducibility run {run} hashes equal")

def random_test():
    upyc.cuda_set_sync_mode(True, True)
    #img.display("newly created")
    img_scaled = img.scale(1280, 720)
    #img_scaled.display("scaled")
    #img_converted = img.convert(upyc.YUV420_DEVICE)
    #img_converted.display("converted")
    #img_converted2 = img_converted.convert(upyc.NV12_DEVICE)
    #img_converted2.display("converted2")

    # Convert back to BGR for OpenCV display
    print(img.hash(), img_scaled.hash())
    round_trip_bgr = cv2.cvtColor(img_scaled.to_numpy(), cv2.COLOR_RGB2BGR)

    # Display the image
    cv2.imshow("Round-trip image", round_trip_bgr)
    cv2.waitKey(10)
    cv2.waitKey(10)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    # run tensortRT inference
    inf = upyc.c_infer("/mldata/weights/trt/yolo11l-dpa-131224.trt")
    dets=inf.run(img_scaled)
    print(dets)

    # create a NVDEC video decoder
    decoder = upyc.c_decoder()
    with open("/mldata/video/MOT20-01.264", "rb") as f:
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
        dets=inf.run(frame)
        #print(dets)
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
                #print(x,y,costs[x][y],dx,dy)
                stop=[start[0]+flow[y][x][0], start[1]+flow[y][x][1]]
                if abs(dx)>0.001 or abs(dy)>0.001:
                    d.draw_line(start, stop, clr=(255,255,255,255))
        d.show(arr, is_rgb=True)
        d.get_events(30)
# Load JPEG image using OpenCV
bgr_img = cv2.imread("/mldata/image/arrest.jpg")
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

img = upyc.c_image.from_numpy(rgb_img)

test_hash(rgb_img)
test_reproducibility(rgb_img)


