import cv2
import numpy as np


class RolloutEnsembleBase:
    def __init__(self):
        self.rollout_instances = []
        self.env = None
        self.key = None

    def setup_env(self, OperationEnvClass, **config):
        render_mode = None if config.get("no_render", False) else "human"
        try:
            op_inst = OperationEnvClass()
        except TypeError:
            op_inst = OperationEnvClass(**config)
        try:
            op_inst.setup_env(render_mode=render_mode)
        except TypeError:
            op_inst.setup_env(render_mode=render_mode, **config)
        self._operation_template = op_inst
        self.env = getattr(op_inst, "env", None)
        if self.env is None:
            raise RuntimeError(
                f"Operation class {OperationEnvClass.__name__}.setup_env() did not set `self.env`."
            )
        return self.env

    def set_rollout_inst_list(self, rollout_instances):
        self.rollout_instances = rollout_instances

    def init_vars(self):
        for rollout_instance in self.rollout_instances:
            rollout_instance.reset_flag = True
            rollout_instance.quit_flag = False
            rollout_instance.inference_duration_list = []

    def fetch_env_commands(self, env_action_parts):
        for rollout_instance in self.rollout_instances:
            cmd_list = rollout_instance.fetch_env_commands()
            if len(cmd_list) > 0:
                cmd_vec = np.concatenate(cmd_list)
            else:
                cmd_vec = np.zeros(0, dtype=np.float64)
            env_action_parts.append(cmd_vec)

    def record_rollout_data(self):
        for rollout_instance in self.rollout_instances:
            if (
                rollout_instance.args.save_rollout
                and rollout_instance.phase_manager.is_phase("RolloutPhase")
            ):
                rollout_instance.record_data()

    def set_obs(self, obs, reward, info):
        for rollout_instance in self.rollout_instances:
            rollout_instance.obs = obs
            rollout_instance.reward = reward
            rollout_instance.info = info

    def post_update(self):
        for rollout_instance in self.rollout_instances:
            rollout_instance.phase_manager.post_update()

    def check_transition(self):
        for rollout_instance in self.rollout_instances:
            try:
                rollout_instance.phase_manager.check_transition()
            except AttributeError:
                pass  # Ignore AttributeError if phase_manager absent; safe as not all rollouts have it

    def save_rollout_results(self):
        for rollout_instance in self.rollout_instances:
            rollout_instance.dump_rollout_result()

            rollout_instance.print_statistics()

    def activate_quit_flag(self):
        for rollout_instance in self.rollout_instances:
            rollout_instance.quit_flag = True

    def is_quit_activated(self):
        for rollout_instance in self.rollout_instances:
            if rollout_instance.quit_flag:
                return True
        return False

    def run(self, **kwargs):
        if self.env is None:
            raise RuntimeError("env not initialized. Call setup_env() before run().")

        if len(self.rollout_instances) == 0:
            raise RuntimeError("No rollout instances set.")

        self.init_vars()

        while True:
            env_action_parts = []
            self.fetch_env_commands(env_action_parts)
            env_action = np.mean(np.stack(env_action_parts, axis=0), axis=0).astype(
                env_action_parts[0].dtype, copy=True
            )

            self.record_rollout_data()

            obs, reward, _, _, info = self.env.step(env_action)
            self.set_obs(obs, reward, info)

            self.post_update()

            self.key = cv2.waitKey(1)
            self.check_transition()

            if self.key == 27:  # escape key
                self.activate_quit_flag()
            if self.is_quit_activated():
                break

        self.save_rollout_results()
