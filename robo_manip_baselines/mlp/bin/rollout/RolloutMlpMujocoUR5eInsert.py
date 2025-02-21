from robo_manip_baselines.common.rollout import RolloutMujocoUR5eInsert
from robo_manip_baselines.mlp import RolloutMlp


class RolloutMlpMujocoUR5eInsert(RolloutMlp, RolloutMujocoUR5eInsert):
    pass


if __name__ == "__main__":
    rollout = RolloutMlpMujocoUR5eInsert()
    rollout.run()
