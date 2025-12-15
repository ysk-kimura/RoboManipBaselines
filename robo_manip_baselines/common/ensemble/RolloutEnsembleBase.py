import cv2


class RolloutEnsembleBase:
    def __init__(self, rollout_instances=None):
        self.rollout_instances = rollout_instances or []

    def set_rollout_inst_list(self, lst):
        self.rollout_instances = lst

    def reset(self, **kwargs):
        for rollout_instance in self.rollout_instances:
            rollout_instance.reset(**kwargs)

    def run(self, **kwargs):
        for rollout_instance in self.rollout_instances:
            rollout_instance.reset_run_vars()

            while True:
                env_action = rollout_instance.get_env_action()
                rollout_instance.record_rollout_data()

                (
                    rollout_instance.obs,
                    rollout_instance.reward,
                    _,
                    _,
                    rollout_instance.info,
                ) = rollout_instance.env.step(env_action)

                rollout_instance.post_phase_update()

                rollout_instance.key = cv2.waitKey(1)
                try:
                    rollout_instance.check_phase_transition()
                except AttributeError:
                    pass  # Ignore AttributeError if phase_manager absent; safe as not all rollouts have it

                if rollout_instance.key == 27:  # escape key
                    rollout_instance.quit_flag = True
                if rollout_instance.quit_flag:
                    break

            rollout_instance.dump_rollout_result()

            rollout_instance.print_statistics()

            # rollout_instance.env.close()
