import os
import re
import time
from abc import ABCMeta, abstractmethod

import cv2
import gymnasium as gym
import numpy as np
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids


class RealEnvBase(gym.Env, metaclass=ABCMeta):
    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        **kwargs,
    ):
        # Setup environment parameters
        self.init_time = time.time()
        self.dt = 0.02  # [s]
        if kwargs.get("scale_dt") is not None:
            self.dt *= kwargs["scale_dt"]

    def setup_realsense(self, camera_ids):
        self.cameras = {}
        detected_camera_ids = get_device_ids()
        for camera_name, camera_id in camera_ids.items():
            if camera_id is None:
                self.cameras[camera_name] = None
                continue

            if camera_id not in detected_camera_ids:
                raise ValueError(
                    f"Specified camera (name: {camera_name}, ID: {camera_id}) not detected. Detected camera IDs: {detected_camera_ids}"
                )

            camera = RealSenseCamera(device_id=camera_id, flip=False)
            frames = camera._pipeline.wait_for_frames()
            color_intrinsics = (
                frames.get_color_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.color_fovy = np.rad2deg(
                2 * np.arctan(color_intrinsics.height / (2 * color_intrinsics.fy))
            )
            depth_intrinsics = (
                frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.depth_fovy = np.rad2deg(
                2 * np.arctan(depth_intrinsics.height / (2 * depth_intrinsics.fy))
            )

            self.cameras[camera_name] = camera

    def setup_gelsight(self, tactile_ids):
        self.tactiles = {}

        for tactile_name, tactile_id in tactile_ids.items():
            for device_name in os.listdir("/sys/class/video4linux"):
                real_device_name = os.path.realpath(
                    "/sys/class/video4linux/" + device_name + "/name"
                )
                with (
                    open(real_device_name, "rt") as device_name_file
                ):  # "rt": read-text mode ("t" is default, so "r" alone is the same)
                    detected_tactile_id = device_name_file.read().rstrip()
                is_found = tactile_id in detected_tactile_id
                print(
                    "{} {} -> {}".format(
                        "FOUND!" if is_found else "      ",
                        device_name,
                        detected_tactile_id,
                    )
                )
                if not is_found:
                    continue
                tactile_num = int(re.search("\d+$", device_name).group(0))
                tactile = cv2.VideoCapture(tactile_num)
                if tactile is None or not tactile.isOpened():
                    print(f"Warning: unable to open video source: {tactile_num}")
                    continue
                self.tactiles[tactile_name] = tactile
                break

    def reset(self, *, seed=None, options=None):
        self.init_time = time.time()

        super().reset(seed=seed)

        self._reset_robot()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._set_action(action, duration=self.dt, joint_vel_limit_scale=2.0, wait=True)

        observation = self._get_obs()
        reward = 0.0
        terminated = False
        info = self._get_info()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def close(self):
        pass

    @abstractmethod
    def _reset_robot(self):
        pass

    @abstractmethod
    def _set_action(self):
        pass

    @abstractmethod
    def _get_obs(self):
        pass

    def _get_info(self):
        info = {}

        if len(self.camera_names) + len(self.tactile_names) == 0:
            return info

        # Get images
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

            rgb_image, depth_image = camera.read((640, 480))
            info["rgb_images"][camera_name] = rgb_image
            info["depth_images"][camera_name] = 1e-3 * depth_image[:, :, 0]  # [m]

        for tactile_name, tactile in self.tactiles.items():
            ret, rgb_image = tactile.read()
            assert ret, "ERROR! reading image from tactile!"
            info["rgb_images"][tactile_name] = rgb_image
            info["depth_images"][tactile_name] = None
            continue

        return info

    def get_joint_pos_from_obs(self, obs):
        """Get joint position from observation."""
        return obs["joint_pos"]

    def get_joint_vel_from_obs(self, obs):
        """Get joint velocity from observation."""
        return obs["joint_vel"]

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (fx, fy, fz, nx, ny, nz) from observation."""
        return obs["wrench"]

    def get_time(self):
        """Get real-world time. [s]"""
        return time.time() - self.init_time

    @property
    def camera_names(self):
        """Camera names being measured."""
        return self.cameras.keys()

    @property
    def tactile_names(self):
        return self.tactiles.keys()

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        camera = self.cameras[camera_name]
        if camera is None:
            return 45.0  # dummy
        return camera.depth_fovy

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        raise NotImplementedError("[RealEnvBase] modify_world is not implemented.")

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        # In a real-world environment, it is not possible to programmatically draw markers
        pass
