from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCabinet
from robo_manip_baselines.mlp import RolloutMlp


class RolloutMlpMujocoUR5eCabinet(RolloutMlp, RolloutMujocoUR5eCabinet):
    pass


if __name__ == "__main__":
    rollout = RolloutMlpMujocoUR5eCabinet()
    rollout.run()
