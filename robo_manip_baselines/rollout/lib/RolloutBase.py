import argparse
import os
import pickle
import sys
import time
from abc import ABCMeta, abstractmethod

import cv2
import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchvision.transforms import v2

from robo_manip_baselines.common import (
    DataKey,
    MotionManager,
    Phase,
    PhaseManager,
    PhaseOrder,
    normalize_data,
)


class RolloutBase(metaclass=ABCMeta):
    def __init__(self):
        self.setup_args()

        self.setup_model_meta_info()

        self.setup_policy()

        self.setup_env()

        self.setup_plot()

        self.setup_variables()

        # Setup motion manager
        self.motion_manager = MotionManager(self.env)

        # Setup phase manager
        self.phase_manager = PhaseManager(self.env, PhaseOrder.ROLLOUT)

        # Setup environment
        self.env.unwrapped.world_random_scale = self.args.world_random_scale
        self.env.unwrapped.modify_world(world_idx=self.args.world_idx)

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--checkpoint", type=str, required=True, help="checkpoint file"
        )

        parser.add_argument(
            "--world_idx",
            type=int,
            default=0,
            help="index of the simulation world (0-5)",
        )
        parser.add_argument(
            "--world_random_scale",
            nargs="+",
            type=float,
            help="random scale of simulation world (no randomness by default)",
        )
        parser.add_argument(
            "--skip",
            type=int,
            help="step interval to infer policy",
        )
        parser.add_argument(
            "--skip_draw",
            type=int,
            help="step interval to draw the plot",
        )
        parser.add_argument(
            "--scale_dt",
            type=float,
            help="dt scale of environment (used only in real-world environments)",
        )
        parser.add_argument("--seed", type=int, default=42, help="random seed")
        parser.add_argument(
            "--win_xy_policy",
            type=int,
            nargs=2,
            help="xy position of window to plot policy information",
        )
        parser.add_argument(
            "--wait_before_start",
            action="store_true",
            help="whether to wait a key input before starting motion",
        )

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

        if self.args.world_random_scale is not None:
            self.args.world_random_scale = np.array(self.args.world_random_scale)

    def setup_model_meta_info(self):
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")
        with open(model_meta_info_path, "rb") as f:
            self.model_meta_info = pickle.load(f)
        print(
            f"[{self.__class__.__name__}] Load model meta info: {model_meta_info_path}"
        )

        # Set state and action information
        self.state_keys = self.model_meta_info["state"]["keys"]
        self.action_keys = self.model_meta_info["action"]["keys"]
        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.state_dim = len(self.model_meta_info["state"]["example"])
        self.action_dim = len(self.model_meta_info["action"]["example"])

        # Set skip if not specified
        if self.args.skip is None:
            self.args.skip = self.model_meta_info["data"]["skip"]
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    @abstractmethod
    def setup_policy(self):
        pass

    @abstractmethod
    def setup_env(self):
        pass

    def setup_plot(self, fig_ax=None):
        matplotlib.use("agg")

        if fig_ax is None:
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(13.5, 6.0), dpi=60, squeeze=False
            )
        else:
            self.fig, self.ax = fig_ax

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()
        cv2.imshow(
            "Policy image",
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

        if self.args.win_xy_policy is not None:
            cv2.moveWindow("Policy image", *self.args.win_xy_policy)
        cv2.waitKey(1)

        self.action_plot_scale = np.concatenate(
            [
                DataKey.get_plot_scale(action_key, self.env)
                for action_key in self.action_keys
            ]
        )

    def setup_variables(self):
        self.image_transforms = v2.ToDtype(torch.float32, scale=True)
        self.policy_action_list = np.empty((0, self.action_dim))

    def print_policy_info(self):
        print(
            f"[{self.__class__.__name__}] Construct {self.policy_name} policy.\n"
            f"  - state dim: {self.state_dim}, action dim: {self.action_dim}, camera num: {len(self.camera_names)}\n"
            f"  - state keys: {self.state_keys}\n"
            f"  - action keys: {self.action_keys}\n"
            f"  - camera names: {self.camera_names}\n"
            f"  - skip: {self.args.skip}"
        )

    def load_ckpt(self):
        print(f"[{self.__class__.__name__}] Load {self.args.checkpoint}")
        self.policy.load_state_dict(torch.load(self.args.checkpoint, weights_only=True))
        self.policy.cuda()
        self.policy.eval()

    def run(self):
        self.obs, self.info = self.env.reset(seed=self.args.seed)

        inference_duration_list = []
        while True:
            if self.phase_manager.phase == Phase.ROLLOUT:
                if self.rollout_time_idx % self.args.skip == 0:
                    inference_start_time = time.time()
                    self.infer_policy()
                    inference_duration_list.append(time.time() - inference_start_time)

            self.set_command()

            env_action = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)
            self.obs, _, _, _, self.info = self.env.step(env_action)

            if self.phase_manager.phase == Phase.ROLLOUT:
                if self.rollout_time_idx % self.args.skip_draw == 0:
                    self.draw_plot()

            # Manage phase
            key = cv2.waitKey(1)
            if self.phase_manager.phase == Phase.INITIAL:
                initial_duration = 1.0  # [s]
                if (
                    not self.args.wait_before_start
                    and self.phase_manager.get_phase_elapsed_duration()
                    > initial_duration
                ) or (self.args.wait_before_start and key == ord("n")):
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.PRE_REACH:
                pre_reach_duration = 0.7  # [s]
                if self.phase_manager.get_phase_elapsed_duration() > pre_reach_duration:
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.REACH:
                reach_duration = 0.3  # [s]
                if self.phase_manager.get_phase_elapsed_duration() > reach_duration:
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.GRASP:
                grasp_duration = 0.5  # [s]
                if self.phase_manager.get_phase_elapsed_duration() > grasp_duration:
                    self.rollout_time_idx = 0
                    print(
                        f"[{self.__class__.__name__}] Press the 'n' key to finish policy rollout."
                    )
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.ROLLOUT:
                self.rollout_time_idx += 1
                if key == ord("n"):
                    print(f"[{self.__class__.__name__}] Statistics on policy inference")
                    policy_model_size = self.calc_model_size()
                    print(
                        f"  - Policy model size [MB] | {policy_model_size / 1024**2:.2f}"
                    )
                    gpu_memory_usage = torch.cuda.max_memory_reserved()
                    print(
                        f"  - GPU memory usage [GB] | {gpu_memory_usage / 1024**3:.3f}"
                    )
                    inference_duration_list = np.array(inference_duration_list)
                    print(
                        "  - Inference duration [s] | "
                        f"mean: {inference_duration_list.mean():.2e}, std: {inference_duration_list.std():.2e} "
                        f"min: {inference_duration_list.min():.2e}, max: {inference_duration_list.max():.2e}"
                    )
                    print(f"[{self.__class__.__name__}] Press the 'n' key to exit.")
                    self.phase_manager.set_next_phase()
            elif self.phase_manager.phase == Phase.END:
                if key == ord("n"):
                    break
            if key == 27:  # escape key
                break

        # self.env.close()

    @abstractmethod
    def infer_policy(self):
        pass

    def get_state(self):
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_measured_data(state_key, self.obs)
                    for state_key in self.state_keys
                ]
            )

        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state[np.newaxis], dtype=torch.float32).cuda()

        return state

    def get_images(self):
        images = np.stack(
            [self.info["rgb_images"][camera_name] for camera_name in self.camera_names],
            axis=0,
        )

        images = np.einsum("k h w c -> k c h w", images)
        images = torch.tensor(images, dtype=torch.uint8)
        images = self.image_transforms(images)[np.newaxis].cuda()

        return images

    def set_command(self):
        if self.phase_manager.phase == Phase.ROLLOUT:
            is_skip = self.rollout_time_idx % self.args.skip != 0
            action_idx = 0
            for action_key in self.action_keys:
                action_dim = DataKey.get_dim(action_key, self.env)
                self.motion_manager.set_command_data(
                    action_key,
                    self.policy_action[action_idx : action_idx + action_dim],
                    is_skip,
                )
                action_idx += action_dim
        else:
            self.set_arm_command()
            self.set_gripper_command()

    def set_arm_command(self):
        pass

    def set_gripper_command(self):
        if self.phase_manager.phase == Phase.GRASP:
            self.motion_manager.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS,
                self.env.action_space.high[self.env.unwrapped.gripper_joint_idxes],
            )

    @abstractmethod
    def draw_plot(self):
        pass

    def plot_images(self, axes):
        for camera_idx, camera_name in enumerate(self.camera_names):
            axes[camera_idx].imshow(self.info["rgb_images"][camera_name])
            axes[camera_idx].set_title(f"{camera_name} image", fontsize=20)

    def plot_action(self, ax):
        history_size = 100
        ax.plot(self.policy_action_list[-1 * history_size :] * self.action_plot_scale)
        ax.set_title("scaled action", fontsize=20)
        ax.set_xlabel("step", fontsize=16)
        ax.set_xlim(0, history_size - 1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.axis("on")

    def calc_model_size(self):
        # https://discuss.pytorch.org/t/finding-model-size/130275/2
        param_size = 0
        for param in self.policy.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.policy.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
