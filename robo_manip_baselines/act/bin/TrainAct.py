import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from detr.models.detr_vae import DETRVAE
from policy import ACTPolicy

from robo_manip_baselines.act import RmbActDataset
from robo_manip_baselines.common import TrainBase


class TrainAct(TrainBase):
    policy_name = "ACT"
    policy_dir = os.path.join(os.path.dirname(__file__), "..")
    DatasetClass = RmbActDataset

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.set_defaults(batch_size=8)
        parser.set_defaults(num_epochs=1000)
        parser.set_defaults(lr=1e-5)

        parser.add_argument("--kl_weight", type=int, default=10, help="KL weight")
        parser.add_argument(
            "--chunk_size", type=int, default=100, help="action chunking size"
        )
        parser.add_argument(
            "--hidden_dim", type=int, default=512, help="hidden dimension"
        )
        parser.add_argument(
            "--dim_feedforward", type=int, default=3200, help="feedforward dimension"
        )

        super().setup_args(parser)

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["chunk_size"] = self.args.chunk_size

    def setup_policy(self):
        # Set policy args
        policy_args = {
            "lr": self.args.lr,
            "num_queries": self.args.chunk_size,
            "kl_weight": self.args.kl_weight,
            "hidden_dim": self.args.hidden_dim,
            "dim_feedforward": self.args.dim_feedforward,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": self.args.camera_names,
        }
        self.model_meta_info["policy"]["args"] = policy_args

        # Construct policy
        DETRVAE.set_state_dim(len(self.model_meta_info["state"]["example"]))
        DETRVAE.set_action_dim(len(self.model_meta_info["action"]["example"]))
        self.policy = ACTPolicy(policy_args)
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = self.policy.configure_optimizers()

        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {self.args.chunk_size}")

    def train_loop(self):
        best_ckpt_info = {"loss": np.inf}
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    batch_result = self.policy(*[d.cuda() for d in data])
                    batch_result_list.append(self.detach_batch_result(batch_result))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                best_ckpt_info = self.update_best_ckpt(best_ckpt_info, epoch_summary)

            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                batch_result = self.policy(*[d.cuda() for d in data])
                loss = batch_result["loss"]
                loss.backward()
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result(batch_result))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt(best_ckpt_info)


if __name__ == "__main__":
    train = TrainAct()
    train.run()
    train.close()
