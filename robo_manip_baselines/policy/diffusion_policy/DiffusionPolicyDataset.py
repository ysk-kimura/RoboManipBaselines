import cv2
import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    get_skipped_data_seq,
)


class DiffusionPolicyDataset(DatasetBase):
    """Dataset to train diffusion policy."""

    def setup_variables(self):
        # Set chunk_info_list
        self.chunk_info_list = []
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        # Set pad_before and pad_after to values one less than n_obs_steps and n_action_steps, respectively
        # Ref: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/task/pusht_image.yaml#L36-L37
        pad_before = self.model_meta_info["data"]["n_obs_steps"] - 1
        pad_after = self.model_meta_info["data"]["n_action_steps"] - 1

        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                for start_time_idx in range(
                    -1 * pad_before, episode_len - (horizon - 1) + pad_after
                ):
                    self.chunk_info_list.append((episode_idx, start_time_idx))

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + horizon), 0, episode_len - 1
            )

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            image_list = np.stack(
                [
                    rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][time_idxes]
                    for camera_name in self.model_meta_info["image"]["camera_names"]
                ],
                axis=0,
            )

        # Resize images
        image_size_list = self.model_meta_info["data"]["image_size_list"]
        resized_image_list = []
        for image, image_size in zip(image_list, image_size_list):
            resized_image_list.append(
                [cv2.resize(single_image, image_size) for single_image in image]
            )
        image_list = np.array(resized_image_list)

        # Pre-convert data
        state, action, image_list = self.pre_convert_data(state, action, image_list)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(image_list, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Convert to data structure of policy input and output
        data = {"obs": {}, "action": action_tensor}
        if len(self.model_meta_info["state"]["keys"]) > 0:
            data["obs"]["state"] = state_tensor
        for camera_idx, camera_name in enumerate(
            self.model_meta_info["image"]["camera_names"]
        ):
            data["obs"][DataKey.get_rgb_image_key(camera_name)] = images_tensor[
                camera_idx
            ]

        return data
