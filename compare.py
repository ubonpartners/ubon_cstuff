import numpy as np
import matplotlib.pyplot as plt
import av
import numpy as np
from skimage.transform import resize
import ubon_pycstuff.ubon_pycstuff as upyc
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def decode_frame_ref(filepath, n=0):
    container = av.open(filepath, format='h264')
    for i, frame in enumerate(container.decode(video=0)):
        if i == n:
            return frame.to_rgb().to_ndarray()
    raise ValueError(f"Frame index {n} not found (only {i+1} frames available)")


def compare_frames(frames, refs, names):
    diffs=[]
    abs_means=[]
    means=[]
    ssims=[]
    for i,f in enumerate(frames):
        diff_img = (np.abs(f.astype(int) - refs[i].astype(int))*4).astype(np.uint8)
        diffs.append(diff_img)
        abs_mean_diff = np.mean(np.abs(f.astype(np.int16) - refs[i].astype(np.int16)))
        mean_diff = np.mean(f.astype(np.int16) - refs[i].astype(np.int16))
        abs_means.append(abs_mean_diff)
        means.append(mean_diff)
        ssim_value, _ = ssim(f, refs[i], full=True, channel_axis=-1)
        ssims.append(ssim_value)

    # Display the images
    plt.figure(figsize=(15, 10))

    for i,f in enumerate(frames):
        plt.subplot(len(frames), 3, i*3+1)
        plt.title("Frame "+names[i])
        plt.imshow(frames[i])
        plt.axis("off")

        plt.subplot(len(frames), 3, i*3+2)
        plt.title("Ref "+names[i])
        plt.imshow(refs[i])
        plt.axis("off")

        plt.subplot(len(frames), 3, i*3+3)
        plt.title(f"Diff {names[i]} mean {abs_means[i]:5.3f},{ssims[i]:0.3f}")
        plt.imshow(diffs[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def compare_decode():
    #v="/mldata/video/operahouse.264"#
    v="/mldata/video/MOT20-05.264"
    ref=decode_frame_ref(v, 0)
    ref = resize(ref, (720, 1280, 3), order=1, preserve_range=True).astype(np.uint8)
    ref = resize(ref, (360, 552, 3), order=1, preserve_range=True).astype(np.uint8)
    decoder = upyc.c_decoder()
    with open(v, "rb") as f:
        bitstream = f.read()

    # decode some video
    frames = decoder.decode(bitstream)
    frame=frames[0]
    frame=frame.scale(552,360)
    frame=frame.convert(upyc.RGB24_DEVICE)
    frame=frame.to_numpy()

    # Load raw ARGB images
    cevo_rgb_path = "/home/mark/frame_rgb.rgb"
    cevo_yuv_path = "/home/mark/frame_yuv.rgb"
    width, height, channels = 552, 360, 4
    shape = (height, width, channels)
    cevo_rgb = np.fromfile(cevo_rgb_path, dtype=np.uint8).reshape(shape)
    cevo_yuv = np.fromfile(cevo_yuv_path, dtype=np.uint8).reshape(shape)
    cevo_rgb = cevo_rgb[:, :, [3, 0, 1, 2]]
    cevo_yuv = cevo_yuv[:, :, [3, 0, 1, 2]]
    cevo_rgb = cevo_rgb[:, :, 1:]  # Drop alpha (A,R,G,B) → (R,G,B)
    cevo_yuv = cevo_yuv[:, :, 1:]

    compare_frames([frame, cevo_rgb, cevo_yuv], [ref, ref, ref], ["ubonc","cevo rgb", "cevo yuv"])

compare_decode()
