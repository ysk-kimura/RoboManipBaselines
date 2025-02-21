import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import DataKey, Phase
from robo_manip_baselines.teleop import TeleopBase


class TeleopRealXarm7Demo(TeleopBase):
    def __init__(self, robot_ip, camera_ids, gelsight_ids=None):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        super().__init__()

        # Command configuration
        self.gripper_scale = 10.0

    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/RealXarm7DemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
        )
        self.demo_name = self.args.demo_name or "RealXarm7Demo"

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            # No action is required in pre-reach or reach phases
            pass
        else:
            super().set_arm_command()

    def set_gripper_command(self):
        if self.phase_manager.phase == Phase.GRASP:
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS, np.array([800.0])
            )
        else:
            super().set_gripper_command()


if __name__ == "__main__":
    robot_ip = "192.168.1.244"
    camera_ids = {"front": "314422070401", "side": None, "hand": "332522077926"}
    gelsight_ids = {"tactile_left": "GelSight Mini R0B 2D16-V7R5: Ge"}
    teleop = TeleopRealXarm7Demo(robot_ip, camera_ids, gelsight_ids)
    teleop.run()
