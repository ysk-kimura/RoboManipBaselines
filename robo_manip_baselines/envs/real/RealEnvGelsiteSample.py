import os
import re

import cv2
import numpy as np


class RealEnvGelsiteSample:
    def __init__(
        self,
    ):
        pass

    def setup_gelsight(self, camera_ids):
        self.cameras = {}

        # get device ids
        detected_camera_ids = set()
        for device_name in os.listdir("/sys/class/video4linux"):
            real_device_name = os.path.realpath(
                "/sys/class/video4linux/" + device_name + "/name"
            )
            with open(real_device_name, "rt") as device_name_file:
                detected_camera_id = device_name_file.read().rstrip()
                detected_camera_ids.add(detected_camera_id)

        for camera_name, camera_id in camera_ids.items():
            if camera_id not in detected_camera_id:
                raise ValueError(
                    f"Specified camera (name: {camera_name}, ID: {camera_id}) not detected. Detected camera IDs: {detected_camera_ids}"
                )

            camera_num = int(re.search("\d+$", device_name).group(0))
            camera = cv2.VideoCapture(camera_num)
            assert (
                camera is not None and camera.isOpened()
            ), f"Warning: unable to open video source: {camera_num}"

            self.cameras[camera_name] = camera

    def _get_info(self):
        info = {}

        if len(self.camera_names) == 0:
            return info

        # Get camera images
        info["rgb_images"] = {}
        info["depth_images"] = {}
        for camera_name, camera in self.cameras.items():
            if camera is None:
                info["rgb_images"][camera_name] = np.zeros(
                    (480, 640, 3), dtype=np.uint8
                )
                info["depth_images"][camera_name] = np.zeros(
                    (480, 640), dtype=np.float32
                )
                continue

            if type(camera) is cv2.VideoCapture:
                ret, rgb_image = camera.read()
                assert ret, "ERROR! reading image from camera!"
                info["rgb_images"][camera_name] = rgb_image
                info["depth_images"][camera_name] = None
                continue

            rgb_image, depth_image = camera.read((640, 480))
            info["rgb_images"][camera_name] = rgb_image
            info["depth_images"][camera_name] = 1e-3 * depth_image[:, :, 0]  # [m]

        return info

    @property
    def camera_names(self):
        """Camera names being measured."""
        return self.cameras.keys()
