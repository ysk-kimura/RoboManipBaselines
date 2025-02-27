import argparse
import datetime
import sys
import time
from abc import ABCMeta, abstractmethod

import cv2
import matplotlib.pylab as plt
import numpy as np
import pinocchio as pin
import pyspacemouse

from robo_manip_baselines.common import (
    DataKey,
    DataManager,
    MotionManager,
    Phase,
    PhaseManager,
    PhaseOrder,
    convert_depth_image_to_color_image,
    convert_depth_image_to_point_cloud,
    set_random_seed,
)


class TeleopBase(metaclass=ABCMeta):
    MotionManagerClass = MotionManager
    DataManagerClass = DataManager
    PhaseManagerClass = PhaseManager

    def __init__(self):
        # Setup arguments
        self.setup_args()

        set_random_seed(self.args.seed)

        # Setup gym environment
        self.setup_env()
        self.env.reset(seed=self.args.seed)

        # Setup motion manager
        self.motion_manager = self.MotionManagerClass(self.env)

        # Setup data manager
        self.data_manager = self.DataManagerClass(self.env, demo_name=self.demo_name)
        self.data_manager.setup_camera_info()
        self.datetime_now = datetime.datetime.now()

        # Setup phase manager
        self.phase_manager = self.PhaseManagerClass(self.env, PhaseOrder.TELEOP)

        # Setup 3D plot
        if self.args.enable_3d_plot:
            plt.rcParams["keymap.quit"] = ["q", "escape"]
            self.fig, self.ax = plt.subplots(
                len(self.env.unwrapped.camera_names),
                1,
                subplot_kw=dict(projection="3d"),
            )
            self.fig.tight_layout()
            self.point_cloud_scatter_list = [None] * len(
                self.env.unwrapped.camera_names
            )

        # Command configuration
        self._spacemouse_connected = False
        self.command_pos_scale = 1e-2
        self.command_rpy_scale = 0.0  # 5e-3
        self.gripper_scale = 5.0

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        iteration_duration_list = []

        while True:
            iteration_start_time = time.time()

            # Reset
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            # Read spacemouse
            if self.phase_manager.phase == Phase.TELEOP:
                # Empirically, you can call read repeatedly to get the latest device state
                for i in range(10):
                    self.spacemouse_state = pyspacemouse.read()

            # Set command
            self.set_command()

            # Set action
            action = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)

            # Record data
            if (
                self.phase_manager.phase == Phase.TELEOP
                and self.args.replay_log is None
            ):
                self.record_data(obs, info)  # noqa: F821

            # Step environment
            obs, _, _, _, info = self.env.step(action)

            # Draw images
            self.draw_image(info)

            # Draw point clouds
            if self.args.enable_3d_plot:
                self.draw_point_cloud(info)

            # Manage phase
            self.manage_phase()
            if self.quit_flag:
                break

            iteration_duration = time.time() - iteration_start_time
            if self.phase_manager.phase == Phase.TELEOP:
                iteration_duration_list.append(iteration_duration)
            if iteration_duration < self.env.unwrapped.dt:
                time.sleep(self.env.unwrapped.dt - iteration_duration)

        print(f"[{self.__class__.__name__}] Statistics on teleoperation")
        if len(iteration_duration_list) > 0:
            iteration_duration_list = np.array(iteration_duration_list)
            print(
                f"  - Real-time factor | {self.env.unwrapped.dt / iteration_duration_list.mean():.2f}"
            )
            print(
                "  - Iteration duration [s] | "
                f"mean: {iteration_duration_list.mean():.3f}, std: {iteration_duration_list.std():.3f} "
                f"min: {iteration_duration_list.min():.3f}, max: {iteration_duration_list.max():.3f}"
            )

        # self.env.close()

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--demo_name", type=str, default=None, help="demonstration name"
        )
        parser.add_argument(
            "--enable_3d_plot", action="store_true", help="whether to enable 3d plot"
        )
        parser.add_argument(
            "--world_idx_list",
            type=int,
            nargs="*",
            help="list of world indexes (if not given, loop through all world indicies)",
        )
        parser.add_argument(
            "--replay_log",
            type=str,
            default=None,
            help="log file path when replaying log motion",
        )
        parser.add_argument(
            "--replay_keys",
            nargs="+",
            choices=DataKey.COMMAND_DATA_KEYS,
            default=[DataKey.COMMAND_JOINT_POS],
            help="Command data keys when replaying log motion",
        )

        parser.add_argument("--seed", type=int, default=42, help="random seed")

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

    @abstractmethod
    def setup_env(self):
        pass

    def reset(self):
        # Reset managers
        self.motion_manager.reset()
        self.phase_manager.reset()

        # Reset or load data
        if self.args.replay_log is None:
            self.data_manager.reset()
            if self.args.world_idx_list is None:
                world_idx = None
            else:
                world_idx = self.args.world_idx_list[
                    self.data_manager.episode_idx % len(self.args.world_idx_list)
                ]
        else:
            self.data_manager.load_data(self.args.replay_log)
            print(
                f"[{self.__class__.__name__}] Load teleoperation data: {self.args.replay_log}\n"
                f"  - replay keys: {self.args.replay_keys}"
            )
            world_idx = self.data_manager.get_meta_data("world_idx")

            # Set initial joint position for relative command
            if not set(self.args.replay_keys).isdisjoint(
                [DataKey.COMMAND_JOINT_POS_REL, DataKey.COMMAND_EEF_POSE_REL]
            ):
                self.motion_manager.set_command_data(
                    DataKey.COMMAND_JOINT_POS,
                    self.data_manager.get_single_data(DataKey.COMMAND_JOINT_POS, 0),
                )

        # Reset environment
        self.data_manager.setup_env_world(world_idx)
        obs, info = self.env.reset()

        print(
            "[{}] episode_idx: {}, world_idx: {}".format(
                self.demo_name,
                self.data_manager.episode_idx,
                self.data_manager.world_idx,
            )
        )
        print(
            f"[{self.__class__.__name__}] Press the 'n' key to start automatic grasping."
        )

    def set_command(self):
        if self.args.replay_log is not None and self.phase_manager.phase in (
            Phase.TELEOP,
            Phase.END,
        ):
            for replay_key in self.args.replay_keys:
                self.motion_manager.set_command_data(
                    replay_key,
                    self.data_manager.get_single_data(replay_key, self.teleop_time_idx),
                )
        else:
            self.set_arm_command()
            self.set_gripper_command()
            self.motion_manager.draw_markers()

    def set_arm_command(self):
        if self.phase_manager.phase == Phase.TELEOP:
            delta_pos = self.command_pos_scale * np.array(
                [
                    -1.0 * self.spacemouse_state.y,
                    self.spacemouse_state.x,
                    self.spacemouse_state.z,
                ]
            )
            delta_rpy = self.command_rpy_scale * np.array(
                [
                    -1.0 * self.spacemouse_state.roll,
                    -1.0 * self.spacemouse_state.pitch,
                    -2.0 * self.spacemouse_state.yaw,
                ]
            )

            target_se3 = self.motion_manager.target_se3.copy()
            target_se3.translation += delta_pos
            target_se3.rotation = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation

            self.motion_manager.set_command_data(DataKey.COMMAND_EEF_POSE, target_se3)

    def set_gripper_command(self):
        if self.phase_manager.phase == Phase.GRASP:
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS,
                self.env.action_space.high[self.env.unwrapped.gripper_joint_idxes],
            )
        elif self.phase_manager.phase == Phase.TELEOP:
            gripper_joint_pos = self.motion_manager.get_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS
            )
            if (
                self.spacemouse_state.buttons[0] > 0
                and self.spacemouse_state.buttons[-1] <= 0
            ):
                gripper_joint_pos += self.gripper_scale
            elif (
                self.spacemouse_state.buttons[-1] > 0
                and self.spacemouse_state.buttons[0] <= 0
            ):
                gripper_joint_pos -= self.gripper_scale
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS, gripper_joint_pos
            )

    def record_data(self, obs, info):
        # Add time
        self.data_manager.append_single_data(
            DataKey.TIME, self.phase_manager.get_phase_elapsed_duration()
        )

        # Add measured data
        for key in (
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.MEASURED_EEF_POSE,
            DataKey.MEASURED_EEF_WRENCH,
        ):
            self.data_manager.append_single_data(
                key, self.motion_manager.get_measured_data(key, obs)
            )

        # Add command data
        for key in (
            DataKey.COMMAND_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
            DataKey.COMMAND_EEF_POSE,
        ):
            self.data_manager.append_single_data(
                key, self.motion_manager.get_command_data(key)
            )

        # Add relative data
        for key in (
            DataKey.MEASURED_JOINT_POS_REL,
            DataKey.COMMAND_JOINT_POS_REL,
            DataKey.MEASURED_EEF_POSE_REL,
            DataKey.COMMAND_EEF_POSE_REL,
        ):
            self.data_manager.append_single_data(
                key, self.data_manager.calc_rel_data(key)
            )

        # Add image
        for camera_name in self.env.unwrapped.camera_names:
            self.data_manager.append_single_data(
                DataKey.get_rgb_image_key(camera_name),
                info["rgb_images"][camera_name],
            )
            self.data_manager.append_single_data(
                DataKey.get_depth_image_key(camera_name),
                info["depth_images"][camera_name],
            )
        for tactile_name in self.env.unwrapped.tactile_names:
            self.data_manager.append_single_data(
                DataKey.get_rgb_image_key(tactile_name),
                info["rgb_images"][tactile_name],
            )

    def draw_image(self, info):
        phase_image = self.phase_manager.get_phase_image()
        rgb_images = []
        depth_images = []
        for camera_name in self.env.unwrapped.camera_names:
            rgb_image = info["rgb_images"][camera_name]
            image_ratio = rgb_image.shape[1] / rgb_image.shape[0]
            resized_image_width = phase_image.shape[1] / 2
            resized_image_size = (
                int(resized_image_width),
                int(resized_image_width / image_ratio),
            )
            rgb_images.append(cv2.resize(rgb_image, resized_image_size))
            depth_image = convert_depth_image_to_color_image(
                info["depth_images"][camera_name]
            )
            depth_images.append(cv2.resize(depth_image, resized_image_size))
        window_image = cv2.vconcat(
            (
                cv2.hconcat((cv2.vconcat(rgb_images), cv2.vconcat(depth_images))),
                phase_image,
            )
        )
        cv2.namedWindow(
            "image",
            flags=(cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL),
        )
        cv2.imshow("image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))

    def draw_point_cloud(self, info):
        far_clip_list = (3.0, 3.0, 0.8)  # [m]
        for camera_idx, camera_name in enumerate(self.env.unwrapped.camera_names):
            point_cloud_skip = 10
            small_depth_image = info["depth_images"][camera_name][
                ::point_cloud_skip, ::point_cloud_skip
            ]
            small_rgb_image = info["rgb_images"][camera_name][
                ::point_cloud_skip, ::point_cloud_skip
            ]
            fovy = self.data_manager.get_meta_data(
                DataKey.get_depth_image_key(camera_name) + "_fovy"
            )
            xyz_array, rgb_array = convert_depth_image_to_point_cloud(
                small_depth_image,
                fovy=fovy,
                rgb_image=small_rgb_image,
                far_clip=far_clip_list[camera_idx],
            )
            if self.point_cloud_scatter_list[camera_idx] is None:

                def get_min_max(v_min, v_max):
                    return (
                        0.75 * v_min + 0.25 * v_max,
                        0.25 * v_min + 0.75 * v_max,
                    )

                self.ax[camera_idx].view_init(elev=-90, azim=-90)
                self.ax[camera_idx].set_xlim(
                    *get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max())
                )
                self.ax[camera_idx].set_ylim(
                    *get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max())
                )
                self.ax[camera_idx].set_zlim(
                    *get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max())
                )
            else:
                self.point_cloud_scatter_list[camera_idx].remove()
            self.ax[camera_idx].axis("off")
            self.ax[camera_idx].set_box_aspect(np.ptp(xyz_array, axis=0))
            self.point_cloud_scatter_list[camera_idx] = self.ax[camera_idx].scatter(
                xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array
            )
        plt.draw()
        plt.pause(0.001)

    def manage_phase(self):
        key = cv2.waitKey(1)
        if self.phase_manager.phase == Phase.INITIAL:
            if key == ord("n"):
                self.phase_manager.set_next_phase()
        elif self.phase_manager.phase == Phase.PRE_REACH:
            pre_reach_duration = 0.7  # [s]
            if self.phase_manager.get_phase_elapsed_duration() > pre_reach_duration:
                self.phase_manager.set_next_phase()
        elif self.phase_manager.phase == Phase.REACH:
            reach_duration = 0.3  # [s]
            if self.phase_manager.get_phase_elapsed_duration() > reach_duration:
                print(
                    f"[{self.__class__.__name__}] Press the 'n' key to start teleoperation after the gripper is closed."
                )
                self.phase_manager.set_next_phase()
        elif self.phase_manager.phase == Phase.GRASP:
            if key == ord("n"):
                # Setup spacemouse
                if (self.args.replay_log is None) and (not self._spacemouse_connected):
                    self._spacemouse_connected = True
                    pyspacemouse.open()
                self.teleop_time_idx = 0
                if self.args.replay_log is None:
                    print(
                        f"[{self.__class__.__name__}] Press the 'n' key to finish teleoperation."
                    )
                else:
                    print(
                        f"[{self.__class__.__name__}] Start to replay the log motion."
                    )
                self.phase_manager.set_next_phase()
        elif self.phase_manager.phase == Phase.TELEOP:
            self.teleop_time_idx += 1
            if self.args.replay_log is None:
                if key == ord("n"):
                    print(
                        f"[{self.__class__.__name__}] Press the 's' key if the teleoperation succeeded,"
                        " or the 'f' key if it failed. (duration: {:.1f} [s])".format(
                            self.phase_manager.get_phase_elapsed_duration()
                        )
                    )
                    self.phase_manager.set_next_phase()
            else:
                if self.teleop_time_idx == len(
                    self.data_manager.get_data_seq(DataKey.TIME)
                ):
                    self.teleop_time_idx -= 1
                    print(
                        f"[{self.__class__.__name__}] The log motion has finished replaying. Press the 'n' key to exit."
                    )
                    self.phase_manager.set_next_phase()
        elif self.phase_manager.phase == Phase.END:
            if self.args.replay_log is None:
                if key == ord("s"):
                    # Save data
                    self.save_data()
                    self.reset_flag = True
                elif key == ord("f"):
                    print(
                        f"[{self.__class__.__name__}] Teleoperation failed: Reset without saving"
                    )
                    self.reset_flag = True
            else:
                if key == ord("n"):
                    self.quit_flag = True
        if key == 27:  # escape key
            self.quit_flag = True

    def save_data(self, filename=None):
        if filename is None:
            filename = "teleop_data/{}_{:%Y%m%d_%H%M%S}/env{:0>1}/{}_env{:0>1}_{:0>3}.hdf5".format(
                self.demo_name,
                self.datetime_now,
                self.data_manager.world_idx,
                self.demo_name,
                self.data_manager.world_idx,
                self.data_manager.episode_idx,
            )
        self.data_manager.save_data(filename)
        print(
            f"[{self.__class__.__name__}] Teleoperation succeeded: Save the data as {filename}"
        )
