import argparse
import os

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
from eipl.model import SARNN
from eipl.utils import (
    deprocess_img,
    normalization,
    resize_img,
    restore_args,
    tensor2numpy,
)

from robo_manip_baselines.common import DataKey
from robo_manip_baselines.rollout import RolloutBase


class RolloutSarnn(RolloutBase):
    def setup_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--checkpoint",
            type=str,
            help="checkpoint file of SARNN (*.pth)",
            required=True,
        )
        parser.add_argument(
            "--cropped_img_size", default=280, type=int, help="size to crop the image"
        )

        super().setup_args(parser)

        if self.args.skip is None:
            self.args.skip = 6
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def setup_policy(self):
        # Restore parameters
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        self.params = restore_args(os.path.join(checkpoint_dir, "args.json"))

        # Set variables
        self.joint_bounds = np.load(os.path.join(checkpoint_dir, "action_bounds.npy"))
        self.joint_dim = self.joint_bounds.shape[-1]
        self.joint_scales = [1.0] * (self.joint_dim - 1) + [0.01]
        self.im_size = 64
        self.v_min_max = [self.params["vmin"], self.params["vmax"]]
        self.policy_action_list = np.empty((0, self.joint_dim))
        self.rnn_state = None

        # Define model
        self.policy = SARNN(
            rec_dim=self.params["rec_dim"],
            joint_dim=self.joint_dim,
            k_dim=self.params["k_dim"],
            heatmap_size=self.params["heatmap_size"],
            temperature=self.params["temperature"],
            im_size=[self.im_size, self.im_size],
        )
        if self.params["compile"]:
            self.policy = torch.compile(self.policy)

        # Load weight
        print(f"[RolloutSarnn] Load {self.args.checkpoint}")
        ckpt = torch.load(self.args.checkpoint, map_location=torch.device("cpu"))
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.policy.eval()

    def setup_plot(self):
        fig_ax = plt.subplots(1, 3, figsize=(13.5, 6.0), dpi=60, squeeze=False)
        super().setup_plot(fig_ax=fig_ax)

    def infer_policy(self):
        if self.rollout_time_idx % self.args.skip != 0:
            return False

        # Preprocess
        self.obs_front_image = self.info["rgb_images"]["front"]
        [fro_lef, fro_top] = [
            (self.obs_front_image.shape[ax] - self.args.cropped_img_size) // 2
            for ax in [0, 1]
        ]
        [fro_rig, fro_bot] = [
            (p + self.args.cropped_img_size) for p in [fro_lef, fro_top]
        ]
        self.obs_front_image = self.obs_front_image[fro_lef:fro_rig, fro_top:fro_bot, :]
        self.obs_front_image = resize_img(
            np.expand_dims(self.obs_front_image, 0), (self.im_size, self.im_size)
        )[0]
        front_image_input = self.obs_front_image.transpose(2, 0, 1)
        front_image_input = normalization(front_image_input, (0, 255), self.v_min_max)
        front_image_input = torch.Tensor(np.expand_dims(front_image_input, 0))
        joint_input = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)
        joint_input = normalization(joint_input, self.joint_bounds, self.v_min_max)
        joint_input = torch.Tensor(np.expand_dims(joint_input, 0))

        # Infer
        (
            front_image_output,
            joint_output,
            enc_front_pts_output,
            dec_front_pts_output,
            self.rnn_state,
        ) = self.policy(front_image_input, joint_input, self.rnn_state)

        # Postprocess
        self.pred_front_image = tensor2numpy(front_image_output[0])
        self.pred_front_image = deprocess_img(
            self.pred_front_image, self.params["vmin"], self.params["vmax"]
        )
        self.pred_front_image = self.pred_front_image.transpose(1, 2, 0)
        self.policy_action = tensor2numpy(joint_output[0])
        self.policy_action = normalization(
            self.policy_action, self.v_min_max, self.joint_bounds
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, np.expand_dims(self.policy_action, 0)]
        )
        self.enc_front_pts = (
            tensor2numpy(enc_front_pts_output[0]).reshape(self.params["k_dim"], 2)
            * self.im_size
        )
        self.dec_front_pts = (
            tensor2numpy(dec_front_pts_output[0]).reshape(self.params["k_dim"], 2)
            * self.im_size
        )

        return True

    def draw_plot(self):
        if self.rollout_time_idx % self.args.skip_draw != 0:
            return

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Draw observed image
        self.ax[0, 0].imshow(self.obs_front_image)
        for j in range(self.params["k_dim"]):
            self.ax[0, 0].plot(
                self.enc_front_pts[j, 0], self.enc_front_pts[j, 1], "co", markersize=12
            )
            self.ax[0, 0].plot(
                self.dec_front_pts[j, 0],
                self.dec_front_pts[j, 1],
                "rx",
                markersize=12,
                markeredgewidth=2,
            )
        self.ax[0, 0].set_title("Observed image", fontsize=20)

        # Draw predicted image
        self.ax[0, 1].imshow(self.pred_front_image)
        self.ax[0, 1].set_title("Predicted image", fontsize=20)

        # Plot joint
        xlim = 500 // self.args.skip
        self.ax[0, 2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self.ax[0, 2].set_xlim(0, xlim)
        for joint_idx in range(self.policy_action_list.shape[1]):
            self.ax[0, 2].plot(
                np.arange(self.policy_action_list.shape[0]),
                self.policy_action_list[:, joint_idx] * self.joint_scales[joint_idx],
            )
        self.ax[0, 2].set_xlabel("Step", fontsize=16)
        self.ax[0, 2].set_title("Joint", fontsize=20)
        self.ax[0, 2].tick_params(axis="x", labelsize=16)
        self.ax[0, 2].tick_params(axis="y", labelsize=16)
        self.ax[0, 2].axis("on")

        self.fig.tight_layout()
        self.canvas.draw()
        cv2.imshow(
            "Policy image",
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
