import unittest
import numpy as np
import random
import ubon_pycstuff.ubon_pycstuff as upyc

class TestUbonPyCStuff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb_img = np.random.randint(0, 255, (512, 320, 3), dtype=np.uint8)
        upyc.cuda_set_sync_mode(False, False)

    def test_hash_consistency(self):
        img = upyc.c_image.from_numpy(self.rgb_img).convert(upyc.RGB24_HOST)
        img_device = img.convert(upyc.RGB24_DEVICE)
        img_host2 = img_device.convert(upyc.RGB24_HOST)

        self.assertEqual(img.hash(), img_device.hash(), "Host and device memory hashes should match")
        self.assertEqual(img_host2.hash(), img_device.hash(), "Reconverted host hash should match device")

    def test_reproducibility(self):
        
        img = upyc.c_image.from_numpy(self.rgb_img)
        img = img.convert(upyc.YUV420_DEVICE)

        for run in range(10):
            hashes = []
            for i in range(50):
                test = img
                if run == 0:
                    test = test.convert(upyc.YUV420_DEVICE)
                if run >= 1:
                    test = test.scale(1280, 720)
                if run >= 2:
                    test = test.scale(320, 256)
                if run == 9:
                    test = test.scale(816, 736)
                if run in (3, 4, 5):
                    test = test.convert(upyc.NV12_DEVICE)
                if run in (4, 5):
                    test = test.convert(upyc.YUV420_DEVICE)
                if run == 5:
                    test = test.convert(upyc.YUV420_HOST)
                #if run == 6:
                #    test = test.convert(upyc.RGB24_HOST)
                if run in (7, 8):
                    test = test.convert(upyc.RGB_PLANAR_FP32_DEVICE)
                hashes.append(test.hash())

            with self.subTest(run=run):
                self.assertTrue(all(h == hashes[0] for h in hashes), f"Run {run} hashes should be equal")

    def test_reproducibility2(self):
        img = upyc.c_image.from_numpy(self.rgb_img).convert(upyc.MONO_DEVICE)
        img.sync()
        random.seed(42)
        testsets=[]
        for i in range(100):
            testsets.append({"choices":[random.choice([True, False]) for _ in range(10)],
                            "hashes":[]})
        for run in range(200):
            testn=random.randint(0, len(testsets)-1)
            testp=testsets[testn]
            for i in range(20):
                test = img
                if testp["choices"][0]:
                    test = test.convert(upyc.YUV420_DEVICE)
                if testp["choices"][1]:
                    test = test.convert(upyc.MONO_DEVICE)
                if testp["choices"][2]:
                    test = test.scale(64, 64)
                if testp["choices"][2]:
                    test = test.scale(128, 64)
                if testp["choices"][3]:
                    test = test.scale(320, 320)
                if testp["choices"][4]:
                    test = test.scale(1280, 512)
                if testp["choices"][5]:
                    test = test.convert(upyc.YUV420_DEVICE)
                if testp["choices"][6]:
                    test = test.convert(upyc.MONO_DEVICE)
                if testp["choices"][7]:
                    test = test.blur()
                if testp["choices"][5]:
                   test = test.scale(128, 192)
                testsets[testn]["hashes"].append(test.hash())

            with self.subTest(run=run):
                for i in range(len(testsets)):
                    hashes=testsets[testn]["hashes"]
                    if len(hashes)>0:
                        self.assertTrue(all(h == hashes[0] for h in hashes), f"Test2 {i} hashes should be equal {hashes}")

if __name__ == '__main__':
    unittest.main()
