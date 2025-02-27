import gymnasium as gym

from .RolloutBase import RolloutBase


class RolloutMujocoUR5eCabinet(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eCabinetEnv-v0", render_mode="human"
        )
