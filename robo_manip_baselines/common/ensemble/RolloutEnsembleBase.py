import cv2
import numpy as np


class RolloutEnsembleBase:
    def __init__(self, rollout_instances=None):
        self.rollout_instances = rollout_instances or []

    # def set_rollout_inst_list(self, lst):
    #     self.rollout_instances = lst

    # def reset(self, **kwargs):
    #     for rollout_instance in self.rollout_instances:
    #         rollout_instance.reset(**kwargs)

    def run(self, **kwargs):
        for rollout_instance in self.rollout_instances:
            rollout_instance.reset_run_vars()

        while True:
            env_action_parts = []
            for rollout_instance in self.rollout_instances:
                cmd_list = rollout_instance.fetch_env_commands()
                if len(cmd_list) > 0:
                    cmd_vec = np.concatenate(cmd_list)
                else:
                    cmd_vec = np.zeros(0, dtype=np.float64)
                env_action_parts.append(cmd_vec)
            env_action = np.mean(np.stack(env_action_parts, axis=0), axis=0).astype(
                env_action_parts[0].dtype, copy=True
            )

            for rollout_instance in self.rollout_instances:
                rollout_instance.record_rollout_data()

            obs, reward, _, _, info = self.rollout_instances[0].env.step(env_action)
            for rollout_instance in self.rollout_instances:
                rollout_instance.obs = obs
                rollout_instance.reward = reward
                rollout_instance.info = info

            for rollout_instance in self.rollout_instances:
                rollout_instance.post_phase_update()

            for rollout_instance in self.rollout_instances:
                rollout_instance.key = cv2.waitKey(1)
                try:
                    rollout_instance.check_phase_transition()
                except AttributeError:
                    pass  # Ignore AttributeError if phase_manager absent; safe as not all rollouts have it

            if self.rollout_instances[0].key == 27:  # escape key
                self.rollout_instances[0].quit_flag = True
            if self.rollout_instances[0].quit_flag:
                break

        for rollout_instance in self.rollout_instances:
            rollout_instance.dump_rollout_result()

            rollout_instance.print_statistics()

        # self.rollout_instances[0].env.close()
