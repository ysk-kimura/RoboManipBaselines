import time
from os import path

import numpy as np
from gymnasium.spaces import Box, Dict
from xarm.wrapper import XArmAPI

from ..RealEnvBase import RealEnvBase


class RealXarm7EnvBase(RealEnvBase):
    action_space = Box(
        low=np.deg2rad(
            [
                -2 * np.pi,
                np.deg2rad(-118),
                -2 * np.pi,
                np.deg2rad(-11),
                -2 * np.pi,
                np.deg2rad(-97),
                -2 * np.pi,
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                2 * np.pi,
                np.deg2rad(120),
                2 * np.pi,
                np.deg2rad(225),
                2 * np.pi,
                np.pi,
                2 * np.pi,
                840.0,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip,
        camera_ids,
        gelsight_ids,
        init_qpos,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.gripper_joint_idxes = [7]
        self.arm_joint_idxes = slice(0, 7)
        self.arm_urdf_path = path.join(
            path.dirname(__file__), "../../assets/common/robots/xarm7/xarm7.urdf"
        )
        self.arm_root_pose = None
        self.ik_eef_joint_id = 7
        self.init_qpos = init_qpos
        self.joint_vel_limit = np.deg2rad(180)  # [rad/s]

        # Connect to xArm7
        print("[RealXarm7EnvBase] Start connecting the xArm7.")
        self.robot_ip = robot_ip
        self.xarm_api = XArmAPI(self.robot_ip)
        self.xarm_api.connect()
        self.xarm_api.motion_enable(enable=True)
        self.xarm_api.ft_sensor_enable(1)
        time.sleep(0.2)
        self.xarm_api.ft_sensor_set_zero()
        time.sleep(0.2)
        self.xarm_api.clean_error()
        self.xarm_api.set_mode(6)
        self.xarm_api.set_state(0)
        self.xarm_api.set_collision_sensitivity(1)
        self.xarm_api.clean_gripper_error()
        self.xarm_api.set_gripper_mode(0)
        self.xarm_api.set_gripper_enable(True)
        time.sleep(0.2)
        xarm_code, joint_states = self.xarm_api.get_joint_states(is_radian=True)
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")
        self.arm_joint_pos_actual = joint_states[0]
        print("[RealXarm7EnvBase] Finish connecting the xArm7.")

        # Connect to RealSense
        self.setup_realsense(camera_ids)
        if gelsight_ids is not None:
            self.setup_gelsight(gelsight_ids)

    def close(self):
        self.xarm_api.disconnect()

    def _reset_robot(self):
        print("[RealXarm7EnvBase] Start moving the robot to the reset position.")
        self._set_action(
            self.init_qpos, duration=None, joint_vel_limit_scale=0.1, wait=True
        )
        print("[RealXarm7EnvBase] Finish moving the robot to the reset position.")

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Overwrite duration or joint_pos for safety
        arm_joint_pos_command = action[self.arm_joint_idxes]
        scaled_joint_vel_limit = (
            np.clip(joint_vel_limit_scale, 0.01, 10.0) * self.joint_vel_limit
        )
        if duration is None:
            duration_min, duration_max = 0.1, 10.0  # [s]
            duration = np.clip(
                np.max(
                    np.abs(arm_joint_pos_command - self.arm_joint_pos_actual)
                    / scaled_joint_vel_limit
                ),
                duration_min,
                duration_max,
            )
        else:
            arm_joint_pos_command_overwritten = self.arm_joint_pos_actual + np.clip(
                arm_joint_pos_command - self.arm_joint_pos_actual,
                -1 * scaled_joint_vel_limit * duration,
                scaled_joint_vel_limit * duration,
            )
            # if np.linalg.norm(arm_joint_pos_command_overwritten - arm_joint_pos_command) > 1e-10:
            #     print("[RealXarm7EnvBase] Overwrite joint command for safety.")
            arm_joint_pos_command = arm_joint_pos_command_overwritten

        # Send command to xArm7
        xarm_code = self.xarm_api.set_servo_angle(
            angle=arm_joint_pos_command,
            speed=scaled_joint_vel_limit,
            mvtime=duration,
            is_radian=True,
            wait=False,
        )
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")

        # Send command to xArm gripper
        gripper_pos = action[self.gripper_joint_idxes][0]
        xarm_code = self.xarm_api.set_gripper_position(gripper_pos, wait=False)
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from xArm7
        xarm_code, joint_states = self.xarm_api.get_joint_states(is_radian=True)
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")
        arm_joint_pos = joint_states[0]
        arm_joint_vel = joint_states[1]
        self.arm_joint_pos_actual = arm_joint_pos.copy()

        # Get state from Robotiq gripper
        xarm_code, gripper_pos = self.xarm_api.get_gripper_position()
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")
        gripper_joint_pos = np.array([gripper_pos], dtype=np.float64)
        gripper_joint_vel = np.zeros(1)

        # Get wrench from force sensor
        wrench = np.array(self.xarm_api.get_ft_sensor_data()[1], dtype=np.float64)
        force = wrench[0:3]
        torque = wrench[3:6]

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
        }
