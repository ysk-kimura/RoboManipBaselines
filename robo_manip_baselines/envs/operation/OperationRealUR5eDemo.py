import gymnasium as gym

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.set_target_open()


class OperationRealUR5eDemo:
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
        )

    def get_pre_motion_phases(self):
        return [GraspPhase(self)]
