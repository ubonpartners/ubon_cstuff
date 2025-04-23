import numpy as np
import ubon_pycstuff.ubon_pycstuff as upyc
import cv2
import stuff


def scale_test(jpeg_file):
    upyc.cuda_set_sync_mode(False, False)
    bgr_img = cv2.imread(jpeg_file)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    print(stuff.image_ssim(rgb_img,rgb_img))

    img = upyc.c_image.from_numpy(rgb_img) # will be RGB24_HOST

    print(stuff.image_ssim(rgb_img, img.to_numpy()))
    img1=img.convert(upyc.NV12_DEVICE)
    print(img1.hash(), stuff.image_ssim(rgb_img, img1.to_numpy()))
    print(stuff.image_ssim(rgb_img, img.convert(upyc.YUV420_DEVICE).to_numpy()))
    print(stuff.image_ssim(rgb_img, img.convert(upyc.YUV420_DEVICE).to_numpy()))
    print(stuff.image_ssim(rgb_img, img.convert(upyc.NV12_DEVICE).to_numpy()))
    print(stuff.image_ssim(rgb_img, img.convert(upyc.YUV420_DEVICE).to_numpy()))
    print(stuff.image_ssim(rgb_img, img.scale(1024,760).to_numpy()))

scale_test("/mldata/image/arrest.jpg")