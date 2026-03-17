import os
import sys

import cv2
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

matplotlib.use("TkAgg")

import numpy as np
import yaml
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../Isaac-GR00T"))
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy

from robo_manip_baselines.common import RolloutBase
from robo_manip_baselines.common.data.DataKey import DataKey


class RolloutGr00t(RolloutBase):
    require_task_desc = True

    def setup_policy(self):
        self.gr00t = Gr00tPolicy(
            model_path=self.args.checkpoint,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            device="cuda",
        )
        modality = self.gr00t.get_modality_config()
        self.camera_names = [
            s.replace("_rgb", "") for s in modality["video"].modality_keys
        ]
        self.dataset = LeRobotEpisodeLoader(
            dataset_path="../../../work/Isaac-GR00T/demo_data/mujocour5ecable",
            modality_configs=modality,
            video_backend="torchcodec",
            video_backend_kwargs=None,
        )

        # Print policy information
        self.print_policy_info()

        # self.device = torch.device("cpu")

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names) + 1,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        self.fig, self.ax = fig_ax

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        plt.figure(self.policy_name)

        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()
        plt.imshow(
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR)
        )

        if self.args.win_xy_plot is not None:
            plt.get_current_fig_manager().window.wm_geometry("+20+50")

        if len(self.action_keys) > 0:
            self.action_plot_scale = np.concatenate(
                [DataKey.get_plot_scale(key, self.env) for key in self.action_keys]
            )
        else:
            self.action_plot_scale = np.zeros(0)

    def reset_variables(self):
        super().reset_variables()
        self.policy_action_buf = None

    def setup_model_meta_info(self):
        if "Aloha" in self.env.spec.name:
            self.state_dim = 14
            self.action_dim = 14
        elif "UR5e" in self.env.spec.name:
            self.state_dim = 7
            self.action_dim = 7
        else:
            self.state_dim = 14
            self.action_dim = 14
        self.state_keys = ["measured_joint_pos"]
        self.action_keys = ["command_joint_pos"]

        if self.args.skip is None:
            self.args.skip = 1
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def infer_policy(self):
        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            state = self.get_state()
            state = state[np.newaxis]

            images = self.get_images()

            observation = {
                "state.single_arm": state,
                "annotation.human.task_description": self.args.task_desc,
            }
            for camera_name in self.camera_names:
                observation[f"video.{camera_name}_rgb"] = images[camera_name]
            parse_obs = parse_observation_gr00t(
                observation, self.dataset.modality_configs
            )
            _all_actions, _ = self.gr00t.get_action(parse_obs)
            all_actions = parse_action_gr00t(_all_actions)
            self.policy_action_buf = list(all_actions["action.single_arm"])

        self.policy_action = self.policy_action_buf.pop(0)
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def get_state(self):
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_data(state_key, self.obs)
                    for state_key in self.state_keys
                ],
                dtype=np.float32,
            )

        return state

    def get_images(self):
        # Assume all images are the same size
        images = {}
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name][np.newaxis]
            images[camera_name] = image

        return images

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[0, len(self.camera_names)])

        plt.figure(self.policy_name)

        # Finalize plot
        self.canvas.draw()
        plt.imshow(
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR)
        )

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        self.inference_duration_list = []

        self.motion_manager.reset()

        self.obs, self.info = self.env.reset(seed=self.args.seed)

        self.time = 0
        self.key = 0

        while True:
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            self.phase_manager.pre_update()

            env_action = np.concatenate(
                [
                    self.motion_manager.get_command_data(key)
                    for key in self.env.unwrapped.command_keys_for_step
                ]
            )
            self.obs, self.reward, self.terminated, _, self.info = self.env.step(
                env_action
            )

            self.phase_manager.post_update()

            self.time += 1
            self.phase_manager.check_transition()

            if self.quit_flag:
                break

        if self.args.result_filename is not None:
            print(
                f"[{self.__class__.__name__}] Save the rollout results: {self.args.result_filename}"
            )
            with open(self.args.result_filename, "w") as result_file:
                yaml.dump(self.result, result_file)
