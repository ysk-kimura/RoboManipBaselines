import argparse
import copy
import os
import sys

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../third_party/ManiFlow_Policy/ManiFlow",
    )
)

import torchvision.transforms as v2
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.common.lr_scheduler import get_scheduler
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
from maniflow.policy.maniflow_pointcloud_policy import (
    ManiFlowTransformerPointcloudPolicy,
)

from robo_manip_baselines.common import (
    DataKey,
    TrainBase,
    TrainPointCloudMixin,
)

from .ManiFlowImageDataset import ManiFlowImageDataset
from .ManiFlowPointcloudDataset import ManiFlowPointcloudDataset


class TrainManiFlowPolicy(TrainBase, TrainPointCloudMixin):
    DatasetClass = ManiFlowImageDataset

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(norm_type="limits")

        parser.set_defaults(batch_size=128)
        parser.set_defaults(
            num_epochs=None
        )  # default num_epochs depends on policy_type
        parser.set_defaults(lr=1e-4)
        parser.set_defaults(train_ratio=0.98)

        parser.add_argument(
            "policy_type",
            type=str,
            choices=["Image", "Pointcloud"],
            help="Select whether policy to use (required)",
        )

        parser.add_argument(
            "--weight_decay", type=float, default=1e-3, help="weight decay"
        )

        parser.add_argument(
            "--use_ema",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable or disable exponential moving average (EMA)",
        )

        parser.add_argument(
            "--horizon", type=int, default=16, help="prediction horizon"
        )
        parser.add_argument(
            "--n_obs_steps",
            type=int,
            default=2,
            help="number of steps in the observation sequence to input in the policy",
        )
        parser.add_argument(
            "--n_action_steps",
            type=int,
            default=16,
            help="number of steps in the action sequence to output from the policy",
        )

        parser.add_argument(
            "--encoder_output_dim",
            type=int,
            default=128,
            help="output dimensions of encoder in policy",
        )

        parser.add_argument(
            "--flow_batch_ratio",
            type=float,
            default=0.75,
            help="ratio of data for flow in batch",
        )

        parser.add_argument(
            "--use_pc_color",
            action="store_true",
            help="Whether to use color information in pointcloud-policy",
        )

        parser.add_argument(
            "--visual_condition_length",
            type=int,
            default=128,
            help="length of visual tokens (for pointcloud-policy)",
        )

        parser.add_argument(
            "--image_size",
            type=int,
            nargs=2,
            default=[224, 224],
            help="image size in image-policy",
        )

    def setup_model_meta_info(self):
        # Set default value of num_epochs
        if self.args.policy_type == "Image" and self.args.num_epochs is None:
            self.args.num_epochs = 501
        elif self.args.policy_type == "Pointcloud" and self.args.num_epochs is None:
            self.args.num_epochs = 1010

        super().setup_model_meta_info()

        self.model_meta_info["data"]["horizon"] = self.args.horizon
        self.model_meta_info["data"]["n_obs_steps"] = self.args.n_obs_steps
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps

        if self.args.policy_type == "Pointcloud":
            self.model_meta_info["data"]["use_pc_color"] = self.args.use_pc_color
            num_points, image_size, min_bound, max_bound, rpy_angle = (
                self.setup_pointcloud_info()
            )
            self.model_meta_info["data"]["num_points"] = num_points
            self.model_meta_info["data"]["image_size"] = image_size
            self.model_meta_info["data"]["min_bound"] = min_bound
            self.model_meta_info["data"]["max_bound"] = max_bound
            self.model_meta_info["data"]["rpy_angle"] = rpy_angle
        else:
            self.model_meta_info["data"]["image_size"] = self.args.image_size

        self.model_meta_info["policy"]["use_ema"] = self.args.use_ema
        self.model_meta_info["policy"]["policy_type"] = self.args.policy_type

    def setup_dataset(self):
        if self.args.policy_type == "Pointcloud":
            self.DatasetClass = ManiFlowPointcloudDataset
        return super().setup_dataset()

    def set_data_stats(self):
        super().set_data_stats()
        if self.args.policy_type == "Pointcloud":
            self.set_pointcloud_stats()

    def get_extra_norm_config(self):
        if self.args.norm_type == "limits":
            return {
                "out_min": -1.0,
                "out_max": 1.0,
            }
        else:
            return super().get_extra_norm_config()

    def setup_policy(self):
        # Set policy args
        shape_meta = OmegaConf.create(
            {
                "obs": {},
                "action": {"shape": [len(self.model_meta_info["action"]["example"])]},
            }
        )
        if len(self.args.state_keys) > 0:
            shape_meta["obs"]["state"] = {
                "shape": [len(self.model_meta_info["state"]["example"])],
                "type": "low_dim",
            }
        policy_args = {
            "horizon": self.args.horizon,
            "n_action_steps": self.args.n_action_steps,
            "n_obs_steps": self.args.n_obs_steps,
            "num_inference_steps": 10,
            "obs_as_global_cond": True,
            "diffusion_timestep_embed_dim": 128,
            "diffusion_target_t_embed_dim": 128,
            "visual_cond_len": self.args.visual_condition_length,
            "n_layer": 12,
            "n_head": 8,
            "n_emb": 768,
            "qkv_bias": True,
            "qk_norm": True,
            "flow_batch_ratio": self.args.flow_batch_ratio,
            "consistency_batch_ratio": 1.0 - self.args.flow_batch_ratio,
            "max_lang_cond_len": 1024,
        }
        if self.args.policy_type == "Pointcloud":
            point_dim = 6 if self.args.use_pc_color else 3
            pointcloud_shape = (
                self.model_meta_info["data"]["num_points"],
                point_dim,
            )
            shape_meta["obs"]["point_cloud"] = {
                "shape": pointcloud_shape,
                "type": "point_cloud",
            }
            pointcloud_encoder_conf = OmegaConf.create(
                {
                    "in_channels": point_dim,
                    "out_channels": self.args.encoder_output_dim,
                    "use_layernorm": True,
                    "final_norm": "layernorm",
                    "normal_channel": False,
                    "num_points": self.args.visual_condition_length,
                    "pointwise": True,
                }
            )
            PolicyClass = ManiFlowTransformerPointcloudPolicy
            policy_args.update(
                {
                    "shape_meta": shape_meta,
                    "encoder_output_dim": self.args.encoder_output_dim,
                    "use_pc_color": self.args.use_pc_color,
                    "crop_shape": [80, 80],
                    "pointnet_type": "pointnet",
                    "downsample_points": True,
                    "pointcloud_encoder_cfg": pointcloud_encoder_conf,
                }
            )
        elif self.args.policy_type == "Image":
            for camera_name in self.model_meta_info["image"]["camera_names"]:
                shape_meta["obs"][DataKey.get_rgb_image_key(camera_name)] = {
                    "shape": [3] + self.args.image_size,
                    "horizon": self.args.n_obs_steps,
                    "type": "rgb",
                }
            shape_meta["obs"]["state"]["horizon"] = self.args.n_obs_steps
            shape_meta["action"]["horizon"] = self.args.horizon
            image_transforms = [
                v2.RandomCrop(size=int(0.95 * self.args.image_size[0])),
                v2.RandomRotation(degrees=[-5.0, 5.0]),
                v2.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            ]
            obs_encoder_conf = {
                "shape_meta": shape_meta,
                "model_name": "r3m",
                "pretrained": False,
                "frozen": False,
                "global_pool": "",
                "transforms": image_transforms,
                "feature_aggregation": None,
                "position_encording": "sinusoidal",
                "downsample_ratio": 32,
                "use_group_norm": True,
                "share_rgb_model": False,
                "imagenet_norm": True,
            }
            obs_encoder = TimmObsEncoder(**obs_encoder_conf)
            PolicyClass = ManiFlowTransformerImagePolicy
            policy_args.update(
                {
                    "shape_meta": shape_meta,
                    "visual_cond_len": 1024,
                    "obs_encoder": obs_encoder,
                }
            )

        self.model_meta_info["policy"]["args"] = policy_args

        # Construct policy
        self.policy = PolicyClass(
            **self.model_meta_info["policy"]["args"],
        )

        # Construct exponential moving average (EMA)
        if self.args.use_ema:
            self.ema_policy = copy.deepcopy(self.policy)
            self.ema = EMAModel(
                model=self.ema_policy,
                update_after_step=0,
                inv_gamma=1.0,
                power=0.75,
                min_value=0.0,
                max_value=0.9999,
            )

        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )

        # Transfer to device
        self.policy.cuda()
        if self.args.use_ema:
            self.ema_policy.cuda()
        optimizer_to(self.optimizer, "cuda")

        # Print policy information
        self.print_policy_info()
        print(f"  - policy type: {self.args.policy_type}")
        print(f"  - use ema: {self.args.use_ema}")
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        )
        data_info = self.model_meta_info["data"]
        if self.args.policy_type == "Pointcloud":
            print(
                f"  - with color: {self.args.use_pc_color}, num points: {data_info['num_points']}, image size: {data_info['image_size']}, min bound: {data_info['min_bound']}, max bound: {data_info['max_bound']}, rpy_angle: {data_info['rpy_angle']}"
            )
        else:
            print(f"  - image size: {data_info['image_size']}")

    def train_loop(self):
        ema_model = self.ema_policy if self.args.use_ema else None
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            batch_result_list = []
            for data in self.train_dataloader:
                loss, _ = self.policy.compute_loss(
                    dict_apply(data, lambda x: x.cuda()), ema_model=ema_model
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                if self.args.use_ema:
                    self.ema.step(self.policy)
                batch_result_list.append(
                    self.detach_batch_result(
                        {"loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]}
                    )
                )
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            if self.args.use_ema:
                policy = self.ema_policy
            else:
                policy = self.policy
            policy.eval()
            with torch.inference_mode():
                batch_result_list = []
                for data in self.val_dataloader:
                    loss, _ = policy.compute_loss(
                        dict_apply(data, lambda x: x.cuda()), ema_model=ema_model
                    )
                    batch_result_list.append(self.detach_batch_result({"loss": loss}))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary, policy=policy)
            policy.train()

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>4}", policy=policy)

        # Save last checkpoint
        self.save_current_ckpt("last", policy=policy)

        # Save best checkpoint
        self.save_best_ckpt()
