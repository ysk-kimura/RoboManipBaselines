import unittest

import cv2
import numpy as np
from envs.real.RealEnvGelsiteSample import RealEnvGelsiteSample


class TestRealEnvGelsiteSampleGetInfo(unittest.TestCase):
    def setUp(self):
        self.real_env_gelsite_sample = RealEnvGelsiteSample()
        self.camera_name = "tactile_left"

        # gsrobotics/examples/show3d.py
        #     the device ID can change after unplugging and changing the usb ports.
        #     on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
        camera_ids = {self.camera_name: "GelSight Mini R0B 2D16-V7R5: Ge"}

        self.real_env_gelsite_sample.setup_gelsight(camera_ids)

    def test_real_env_gelsite_sample_get_Info(self):
        info = self.real_env_gelsite_sample._get_info()
        rgb_image = info["rgb_images"][self.camera_name]
        depth_image = info["depth_images"][self.camera_name]

        self.assertIsInstance(rgb_image, np.ndarray)
        self.assertEqual(rgb_image.dtype, np.uint8)
        self.assertEqual(rgb_image.shape[2], 3)  # 3 color channels
        self.assertIsNone(depth_image)
        cv2.imshow("image", rgb_image)

        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    unittest.main()
