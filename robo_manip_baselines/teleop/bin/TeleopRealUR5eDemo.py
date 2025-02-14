import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import Phase
from robo_manip_baselines.teleop import TeleopBase


class TeleopRealUR5eDemo(TeleopBase):
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
        )
        self.demo_name = self.args.demo_name or "RealUR5eDemo"

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


if __name__ == "__main__":
    robot_ip = "192.168.11.4"
    camera_ids = {"front": "145522067924", "side": None, "hand": "153122070885"}
    gelsight_ids = {"tactile_left": "GelSight Mini R0B 2D16-V7R5: Ge"}
    teleop = TeleopRealUR5eDemo(robot_ip, camera_ids, gelsight_ids)
    teleop.run()
