from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy
from robo_manip_baselines.common.rollout import RolloutMujocoXarm7Ring


class RolloutDiffusionPolicyMujocoXarm7Ring(
    RolloutDiffusionPolicy, RolloutMujocoXarm7Ring
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoXarm7Ring()
    rollout.run()
