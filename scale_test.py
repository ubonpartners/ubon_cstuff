import numpy as np
import ubon_pycstuff.ubon_pycstuff as upyc
from PIL import Image
import stuff


def scale_test(jpeg_file):
    upyc.cuda_set_sync_mode(False, False)
    for size in [(640, 480), (352, 288)]:
        img = Image.open(jpeg_file).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        rgb_img = np.asarray(img)

        img = upyc.c_image.from_numpy(rgb_img) # will be RGB24_HOST
        print("Image scale size",size)
        print("SSIM <-> YUV420                     ", stuff.image_ssim(rgb_img, img.convert(upyc.YUV420_DEVICE).to_numpy()))
        print("SSIM <-> NV12                       ", stuff.image_ssim(rgb_img, img.convert(upyc.NV12_DEVICE).to_numpy()))
        print("SSIM <-> RGB24                      ", stuff.image_ssim(rgb_img, img.convert(upyc.RGB24_DEVICE).to_numpy()))
        print("SSIM <-> RGB24 <-> RGB planar fp16  ", stuff.image_ssim(rgb_img, img.convert(upyc.RGB24_DEVICE).convert(upyc.RGB_PLANAR_FP16_DEVICE).to_numpy()))
        print("SSIM <-> RGB24 <-> RGB planar fp32  ", stuff.image_ssim(rgb_img, img.convert(upyc.RGB24_DEVICE).convert(upyc.RGB_PLANAR_FP32_DEVICE).to_numpy()))
        print("SSIM <-> YUV420 <-> RGB planar fp16 ", stuff.image_ssim(rgb_img, img.convert(upyc.YUV420_DEVICE).convert(upyc.RGB_PLANAR_FP16_DEVICE).to_numpy()))
        print("SSIM <-> YUV420 <-> RGB planar fp32 ", stuff.image_ssim(rgb_img, img.convert(upyc.YUV420_DEVICE).convert(upyc.RGB_PLANAR_FP32_DEVICE).to_numpy()))
        print("SSIM scale 1024x760                 ", stuff.image_ssim(rgb_img, img.scale(1024,760).to_numpy()))

scale_test("/mldata/image/arrest.jpg")