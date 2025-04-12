import numpy as np
import ubon_pycstuff.ubon_pycstuff as upyc
import cv2


# Load JPEG image using OpenCV
bgr_img = cv2.imread("/mldata/image/arrest.jpg")
if bgr_img is None:
    raise RuntimeError("Failed to load image")

# Convert from BGR (OpenCV default) to RGB
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

img = upyc.c_image.from_numpy(rgb_img)
img_scaled = img.scale(1024, 768)

inf = upyc.c_infer("/mldata/weights/trt/yolo11l-dpa-131224.trt")

dets=inf.run(img_scaled)
print(dets)
# Convert back to BGR for OpenCV display
round_trip_bgr = cv2.cvtColor(img_scaled.to_numpy(), cv2.COLOR_RGB2BGR)

# Display the image
cv2.imshow("Round-trip image", round_trip_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

decoder = upyc.c_decoder()

with open("/mldata/video/mall_escalators.264", "rb") as f:
    bitstream = f.read()

frames = decoder.decode(bitstream)
print(len(frames))

for i, frame in enumerate(frames):
    arr = frame.to_numpy()
    dets=inf.run(frame)
    print(dets)
    cv2.imshow("Frame", arr)
    cv2.waitKey(30)
