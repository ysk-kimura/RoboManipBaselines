import gymnasium as gym

from robo_manip_baselines.common import DataKey, Phase
from robo_manip_baselines.teleop import TeleopBase


class TeleopMujocoUR5eDoor(TeleopBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eDoorEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoUR5eDoor"

    def set_gripper_command(self):
        if self.phase_manager.phase == Phase.GRASP:
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS,
                self.env.action_space.low[self.env.unwrapped.gripper_joint_idxes],
            )
        else:
            super().set_gripper_command()


if __name__ == "__main__":
    teleop = TeleopMujocoUR5eDoor()
    teleop.run()
