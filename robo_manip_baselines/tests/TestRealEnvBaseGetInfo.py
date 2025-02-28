import unittest

import cv2
import numpy as np

from robo_manip_baselines.envs.real.RealEnvBase import RealEnvBase


class DummyRealEnv(RealEnvBase):
    def __init__(self, gelsight_ids):
        camera_ids = {}
        super().__init__(
            robot_ip=None, camera_ids=camera_ids, gelsight_ids=gelsight_ids
        )
        self.setup_realsense(camera_ids)
        self.setup_gelsight(gelsight_ids)

    def _reset_robot(self, *args, **kwargs):
        pass

    def _set_action(self, *args, **kwargs):
        pass

    def _get_obs(self, *args, **kwargs):
        pass

    @property
    def tactile_names(self):
        return self.tactiles.keys()


class TestRealEnvBaseGetInfo(unittest.TestCase):
    def setUp(self):
        self.tactile_name = "tactile_left"

        # gsrobotics/examples/show3d.py
        #     the device ID can change after unplugging and changing the usb ports.
        #     on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
        gelsight_ids = {self.tactile_name: "GelSight Mini R0B 2D16-V7R5: Ge"}

        self.dummy_real_env = DummyRealEnv(gelsight_ids=gelsight_ids)

    def test_dummy_real_env_get_Info(self):
        info = self.dummy_real_env._get_info()
        rgb_image = info["rgb_images"][self.tactile_name]
        depth_image = info["depth_images"][self.tactile_name]

        self.assertIsInstance(rgb_image, np.ndarray)
        self.assertEqual(rgb_image.dtype, np.uint8)
        self.assertEqual(rgb_image.shape, (600, 800, 3))
        self.assertIsNone(depth_image)

        print("press q on image to exit")
        try:
            while True:
                # get rgb image
                info = self.dummy_real_env._get_info()
                rgb_image = info["rgb_images"][self.tactile_name]
                cv2.imshow("rgb_image", rgb_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("Interrupted!")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    unittest.main()
