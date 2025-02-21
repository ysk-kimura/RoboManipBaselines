import gymnasium as gym

from robo_manip_baselines.teleop import TeleopBase


class TeleopMujocoUR5eCabinet(TeleopBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eCabinetEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoUR5eCabinet"


if __name__ == "__main__":
    teleop = TeleopMujocoUR5eCabinet()
    teleop.run()
