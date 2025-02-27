import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import DataKey, Phase

from .RolloutBase import RolloutBase


class RolloutRealUR5eDemo(RolloutBase):
    def __init__(self, robot_ip, camera_ids, gelsight_ids=None):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        super().__init__()

    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/RealUR5eDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
            scale_dt=self.args.scale_dt,
        )

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            # No action is required in pre-reach or reach phases
            pass
        else:
            super().set_arm_command()

    def set_gripper_command(self):
        if self.phase_manager.phase == Phase.GRASP:
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS, np.array([150.0])
            )
        else:
            super().set_gripper_command()
