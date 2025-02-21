import h5py
import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    get_skipped_single_data,
)


class MlpDataset(DatasetBase):
    """Dataset to train MLP policy."""

    def setup_variables(self):
        skip = self.model_meta_info["data"]["skip"]
        self.accum_step_idxes = []
        for filename in self.filenames:
            with h5py.File(filename, "r") as h5file:
                episode_len = h5file[DataKey.TIME][::skip].shape[0]
                if len(self.accum_step_idxes) == 0:
                    self.accum_step_idxes.append(episode_len)
                else:
                    self.accum_step_idxes.append(
                        self.accum_step_idxes[-1] + episode_len
                    )
        self.accum_step_idxes = np.array(self.accum_step_idxes)

    def __len__(self):
        return self.accum_step_idxes[-1]

    def __getitem__(self, step_idx_in_whole):
        # Set episode_idx and step_idx_in_episode
        skip = self.model_meta_info["data"]["skip"]
        episode_idx = (self.accum_step_idxes > step_idx_in_whole).nonzero()[0][0]
        step_idx_in_episode = step_idx_in_whole
        if episode_idx > 0:
            step_idx_in_episode -= self.accum_step_idxes[episode_idx - 1]

        with h5py.File(self.filenames[episode_idx], "r") as h5file:
            episode_len = h5file[DataKey.TIME][::skip].shape[0]
            assert 0 <= step_idx_in_episode < episode_len

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float32)
            else:
                state = np.concatenate(
                    [
                        get_skipped_single_data(
                            h5file[key], step_idx_in_episode * skip, key, skip
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ]
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_single_data(
                        h5file[key], step_idx_in_episode * skip, key, skip
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ]
            )

            # Load images
            images = np.stack(
                [
                    h5file[DataKey.get_rgb_image_key(camera_name)][
                        step_idx_in_episode * skip
                    ]
                    for camera_name in self.model_meta_info["image"]["camera_names"]
                ],
                axis=0,
            )

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Sort in the order of policy inputs and outputs
        return state_tensor, images_tensor, action_tensor
