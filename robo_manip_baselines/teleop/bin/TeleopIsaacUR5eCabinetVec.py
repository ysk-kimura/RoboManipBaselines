from isaacgym import (  # noqa: I001
    gymapi,  # noqa: F401
    gymtorch,  # noqa: F401
    gymutil,  # noqa: F401
)

import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import DataKey, Phase
from robo_manip_baselines.teleop import TeleopBaseVec


class TeleopIsaacUR5eCabinetVec(TeleopBaseVec):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/IsaacUR5eCabinetEnv-v0",
            num_envs=12,
            render_mode="human",
        )
        self.demo_name = self.args.demo_name or "IsaacUR5eCabinetVec"

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            target_pos = self.env.unwrapped.get_link_pose("ur5e", "base_link")[0:3]
            if self.phase_manager.phase == Phase.PRE_REACH:
                target_pos += np.array([0.33, 0.0, 0.3])  # [m]
            elif self.phase_manager.phase == Phase.REACH:
                target_pos += np.array([0.38, 0.0, 0.3])  # [m]
            target_se3 = pin.SE3(self.motion_manager.target_se3.rotation, target_pos)
            self.motion_manager.set_command_data(DataKey.COMMAND_EEF_POSE, target_se3)
        else:
            super().set_arm_command()


if __name__ == "__main__":
    teleop = TeleopIsaacUR5eCabinetVec()
    teleop.run()
