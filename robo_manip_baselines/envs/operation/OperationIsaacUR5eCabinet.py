import gymnasium as gym

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.set_target_close()


class OperationIsaacUR5eCabinet:
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/IsaacUR5eCabinetEnv-v0",
            render_mode="human",
        )

    def get_pre_motion_phases(self):
        return [
            GraspPhase(self),
        ]
