import os
import sys

import cv2
import matplotlib.pylab as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from detr.models.detr_vae import DETRVAE
from policy import ACTPolicy

from robo_manip_baselines.common import RolloutBase, denormalize_data


class RolloutAct(RolloutBase):
    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {self.model_meta_info['data']['chunk_size']}")

        # Construct policy
        DETRVAE.set_state_dim(self.state_dim)
        DETRVAE.set_action_dim(self.action_dim)
        self.policy = ACTPolicy(self.model_meta_info["policy"]["args"])

        # Register fook to visualize attention images
        def forward_fook(_layer, _input, _output):
            # Output of MultiheadAttention is a tuple (attn_output, attn_output_weights)
            # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            _layer.correlation_mat = _output[1][0].detach().cpu().numpy()

        for layer in self.policy.model.transformer.encoder.layers:
            layer.self_attn.correlation_mat = None
            layer.self_attn.register_forward_hook(forward_fook)

        # Load checkpoint
        self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            max(
                len(self.camera_names) + 1,
                len(self.policy.model.transformer.encoder.layers),
            ),
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

    def setup_variables(self):
        super().setup_variables()

        self.all_actions_history = []

    def infer_policy(self):
        # Infer
        state = self.get_state()
        images = self.get_images()
        all_actions = self.policy(state, images)[0]
        self.all_actions_history.append(
            all_actions.cpu().detach().numpy().astype(np.float64)
        )
        if len(self.all_actions_history) > self.model_meta_info["data"]["chunk_size"]:
            self.all_actions_history.pop(0)

        # Apply temporal ensembling to action
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(self.all_actions_history)))
        exp_weights = exp_weights / exp_weights.sum()
        action = np.zeros(self.action_dim)
        for action_idx, _all_actions in enumerate(reversed(self.all_actions_history)):
            action += exp_weights[::-1][action_idx] * _all_actions[action_idx]
        self.policy_action = denormalize_data(action, self.model_meta_info["action"])
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[0, len(self.camera_names)])

        # Draw attention images
        attention_shape = (15, 20 * len(self.camera_names))
        for layer_idx, layer in enumerate(self.policy.model.transformer.encoder.layers):
            if layer.self_attn.correlation_mat is None:
                continue
            self.ax[1, layer_idx].imshow(
                layer.self_attn.correlation_mat[2:, 1].reshape(attention_shape)
            )
            self.ax[1, layer_idx].set_title(
                f"attention image ({layer_idx})", fontsize=20
            )

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
