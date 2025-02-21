from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCable
from robo_manip_baselines.mlp import RolloutMlp


class RolloutMlpMujocoUR5eCable(RolloutMlp, RolloutMujocoUR5eCable):
    pass


if __name__ == "__main__":
    rollout = RolloutMlpMujocoUR5eCable()
    rollout.run()
