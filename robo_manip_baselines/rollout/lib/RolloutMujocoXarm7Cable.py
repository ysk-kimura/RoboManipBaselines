import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import DataKey, Phase

from .RolloutBase import RolloutBase


class RolloutMujocoXarm7Cable(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoXarm7CableEnv-v0", render_mode="human"
        )

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            target_pos = self.env.unwrapped.get_body_pose("cable_end")[0:3]
            if self.phase_manager.phase == Phase.PRE_REACH:
                target_pos[2] = 1.0  # [m]
            elif self.phase_manager.phase == Phase.REACH:
                target_pos[2] = 0.925  # [m]
            target_se3 = pin.SE3(
                pin.rpy.rpyToMatrix(np.pi, 0.0, -np.pi / 2), target_pos
            )
            self.motion_manager.set_command_data(DataKey.COMMAND_EEF_POSE, target_se3)
        else:
            super().set_arm_command()
