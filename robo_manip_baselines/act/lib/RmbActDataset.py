import h5py
import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    get_skipped_data_seq,
    get_skipped_single_data,
)


class RmbActDataset(DatasetBase):
    """Dataset to train ACT policy."""

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, episode_idx):
        skip = self.model_meta_info["data"]["skip"]
        chunk_size = self.model_meta_info["data"]["chunk_size"]

        with h5py.File(self.filenames[episode_idx], "r") as h5file:
            episode_len = h5file[DataKey.TIME][::skip].shape[0]
            start_time_idx = np.random.choice(episode_len)

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_single_data(
                            h5file[key], start_time_idx * skip, key, skip
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ]
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(
                        h5file[key][start_time_idx * skip :],
                        key,
                        skip,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            images = np.stack(
                [
                    h5file[DataKey.get_rgb_image_key(camera_name)][
                        start_time_idx * skip
                    ]
                    for camera_name in self.model_meta_info["image"]["camera_names"]
                ],
                axis=0,
            )

        # Chunk action
        action_len = action.shape[0]
        action_chunked = np.zeros((chunk_size, action.shape[1]), dtype=np.float64)
        action_chunked[:action_len] = action[:chunk_size]
        is_pad = np.zeros(chunk_size, dtype=bool)
        is_pad[action_len:] = True

        # Pre-convert data
        state, action_chunked, images = self.pre_convert_data(
            state, action_chunked, images
        )

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action_chunked, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)
        is_pad_tensor = torch.tensor(is_pad, dtype=torch.bool)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Sort in the order of policy inputs and outputs
        return state_tensor, images_tensor, action_tensor, is_pad_tensor
