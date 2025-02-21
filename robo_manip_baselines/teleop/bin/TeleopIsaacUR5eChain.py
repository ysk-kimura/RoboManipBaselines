from isaacgym import (  # noqa: I001
    gymapi,  # noqa: F401
    gymtorch,  # noqa: F401
    gymutil,  # noqa: F401
)

import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import DataKey, Phase
from robo_manip_baselines.teleop import TeleopBase


class TeleopIsaacUR5eChain(TeleopBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/IsaacUR5eChainEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "IsaacUR5eChain"

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            target_pos = self.env.unwrapped.get_link_pose("chain_end", "box")[0:3]
            if self.phase_manager.phase == Phase.PRE_REACH:
                target_pos[2] += 0.22  # [m]
            elif self.phase_manager.phase == Phase.REACH:
                target_pos[2] += 0.14  # [m]
            target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
            self.motion_manager.set_command_data(DataKey.COMMAND_EEF_POSE, target_se3)
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
    teleop = TeleopIsaacUR5eChain()
    teleop.run()
