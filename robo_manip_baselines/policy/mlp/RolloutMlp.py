import cv2
import matplotlib.pylab as plt
import numpy as np

from robo_manip_baselines.common import RolloutBase, denormalize_data

from .MlpPolicy import MlpPolicy


class RolloutMlp(RolloutBase):
    def setup_policy(self):
        # Print policy information
        self.print_policy_info()

        # Construct policy
        self.policy = MlpPolicy(
            self.state_dim,
            self.action_dim,
            len(self.camera_names),
            **self.model_meta_info["policy"]["args"],
        )

        # Load checkpoint
        self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names),
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

    def infer_policy(self):
        state = self.get_state()
        images = self.get_images()
        action = self.policy(state, images)[0]
        action = action.cpu().detach().numpy().astype(np.float64)
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
        self.plot_action(self.ax[1, 0])

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
