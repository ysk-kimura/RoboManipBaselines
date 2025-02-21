import argparse
import os

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from robo_manip_baselines.common import TrainBase
from robo_manip_baselines.mlp import MlpDataset, MlpPolicy


class TrainMlp(TrainBase):
    policy_name = "MLP"
    policy_dir = os.path.join(os.path.dirname(__file__), "..")
    DatasetClass = MlpDataset

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.set_defaults(batch_size=32)
        parser.set_defaults(num_epochs=40)
        parser.set_defaults(lr=1e-5)

        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight decay"
        )

        parser.add_argument(
            "--hidden_dim_list",
            type=int,
            nargs="+",
            default=[512, 512],
            help="Dimension list of hidden layers",
        )
        parser.add_argument(
            "--state_feature_dim",
            type=int,
            default=512,
            help="Dimension of state feature",
        )

        super().setup_args(parser)

    def setup_policy(self):
        # Set policy args
        self.model_meta_info["policy"]["args"] = {
            "hidden_dim_list": self.args.hidden_dim_list,
            "state_feature_dim": self.args.state_feature_dim,
        }

        # Construct policy
        self.policy = MlpPolicy(
            len(self.model_meta_info["state"]["example"]),
            len(self.model_meta_info["action"]["example"]),
            len(self.args.camera_names),
            **self.model_meta_info["policy"]["args"],
        )
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        # Print policy information
        self.print_policy_info()

    def train_loop(self):
        best_ckpt_info = {"loss": np.inf}
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    pred_action = self.policy(*[d.cuda() for d in data[0:2]])
                    loss = F.mse_loss(pred_action, data[2].cuda())
                    batch_result_list.append(self.detach_batch_result({"loss": loss}))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                best_ckpt_info = self.update_best_ckpt(best_ckpt_info, epoch_summary)

            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                pred_action = self.policy(*[d.cuda() for d in data[0:2]])
                loss = F.mse_loss(pred_action, data[2].cuda())
                loss.backward()
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result({"loss": loss}))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt(best_ckpt_info)


if __name__ == "__main__":
    train = TrainMlp()
    train.run()
    train.close()
