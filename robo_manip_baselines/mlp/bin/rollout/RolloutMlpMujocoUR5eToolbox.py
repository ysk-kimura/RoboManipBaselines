from robo_manip_baselines.common.rollout import RolloutMujocoUR5eToolbox
from robo_manip_baselines.mlp import RolloutMlp


class RolloutMlpMujocoUR5eToolbox(RolloutMlp, RolloutMujocoUR5eToolbox):
    pass


if __name__ == "__main__":
    rollout = RolloutMlpMujocoUR5eToolbox()
    rollout.run()
