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
infer=upyc.infer_create("/mldata/weights/trt/yolo11l-dpa-131224.trt")
dets=upyc.infer(infer, img_scaled)
print(dets)
# Convert back to BGR for OpenCV display
round_trip_bgr = cv2.cvtColor(img_scaled.to_numpy(), cv2.COLOR_RGB2BGR)

# Display the image
cv2.imshow("Round-trip image", round_trip_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
