import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import DataKey, Phase

from .RolloutBase import RolloutBase


class RolloutMujocoUR5eCloth(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eClothEnv-v0", render_mode="human"
        )

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            target_se3 = pin.SE3(
                pin.rpy.rpyToMatrix(np.pi / 2, 0.0, 0.25 * np.pi),
                self.env.unwrapped.get_body_pose("board")[0:3],
            )
            if self.phase_manager.phase == Phase.PRE_REACH:
                target_se3 *= pin.SE3(
                    pin.rpy.rpyToMatrix(0.0, 0.125 * np.pi, 0.0),
                    np.array([0.0, -0.2, -0.4]),
                )
            elif self.phase_manager.phase == Phase.REACH:
                target_se3 *= pin.SE3(np.identity(3), np.array([0.0, -0.2, -0.3]))
            self.motion_manager.set_command_data(DataKey.COMMAND_EEF_POSE, target_se3)
        else:
            super().set_arm_command()

    def set_gripper_command(self):
        if self.phase_manager.phase == Phase.GRASP:
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS,
                self.env.action_space.low[self.env.unwrapped.gripper_joint_idxes],
            )
        else:
            super().set_gripper_command()
