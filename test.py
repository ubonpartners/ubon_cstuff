import numpy as np
import ubon_pycstuff.ubon_pycstuff as upyc
import cv2
import stuff


# Load JPEG image using OpenCV
bgr_img = cv2.imread("/mldata/image/arrest.jpg")
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

img = upyc.c_image.from_numpy(rgb_img)
img_scaled = img.scale(1024, 768)

# Convert back to BGR for OpenCV display
round_trip_bgr = cv2.cvtColor(img_scaled.to_numpy(), cv2.COLOR_RGB2BGR)

# Display the image
cv2.imshow("Round-trip image", round_trip_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# run tensortRT inference
inf = upyc.c_infer("/mldata/weights/trt/yolo11l-dpa-131224.trt")
dets=inf.run(img_scaled)
print(dets)

# create a NVDEC video decoder
decoder = upyc.c_decoder()
with open("/mldata/video/mall_escalators.264", "rb") as f:
    bitstream = f.read()

# decode some video
frames = decoder.decode(bitstream)

# create an NVOF optical flow engine
flow_engine = upyc.c_nvof(320, 320)

d=stuff.Display(1280,720)

for i, frame in enumerate(frames):
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
    d.show(arr)
    d.get_events(30)
