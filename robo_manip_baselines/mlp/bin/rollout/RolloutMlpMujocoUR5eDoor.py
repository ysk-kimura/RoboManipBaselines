from robo_manip_baselines.common.rollout import RolloutMujocoUR5eDoor
from robo_manip_baselines.mlp import RolloutMlp


class RolloutMlpMujocoUR5eDoor(RolloutMlp, RolloutMujocoUR5eDoor):
    pass


if __name__ == "__main__":
    rollout = RolloutMlpMujocoUR5eDoor()
    rollout.run()
