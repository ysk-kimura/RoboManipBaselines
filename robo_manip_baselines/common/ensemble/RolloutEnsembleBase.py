class RolloutEnsembleBase:
    def __init__(self, rollout_instances=None):
        self.rollout_instances = rollout_instances or []

    def set_rollout_inst_list(self, lst):
        self.rollout_instances = lst

    def reset(self, **kwargs):
        for r in self.rollout_instances:
            r.reset(**kwargs)

    def run(self, **kwargs):
        for r in self.rollout_instances:
            r.run(**kwargs)
