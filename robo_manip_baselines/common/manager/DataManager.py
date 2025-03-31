import os

import h5py
import numpy as np

from robo_manip_baselines import __version__

from ..data.DataKey import DataKey
from ..utils.MathUtils import (
    get_rel_pose_from_se3,
    get_se3_from_pose,
    get_se3_from_rel_pose,
)


class DataManager:
    """Data manager."""

    def __init__(self, env, demo_name=""):
        self.env = env

        self.meta_data = {
            "format": "RmbData-SingleHDF5",
            "demo": demo_name,
            "version": __version__,
        }
        if self.env is not None:
            self.meta_data["env"] = self.env.spec.name

        self.episode_idx = 0
        self.world_idx = 0

        self.reset()

    def reset(self):
        """Reset."""
        self.all_data_seq = {}

    def append_single_data(self, key, data):
        """Append a single data to the data sequence."""
        key = DataKey.replace_deprecated_key(key)  # For backward compatibility
        if key not in self.all_data_seq:
            self.all_data_seq[key] = []
        self.all_data_seq[key].append(data)

    def get_single_data(self, key, time_idx):
        """Get a single data from the data sequence."""
        key = DataKey.replace_deprecated_key(key)  # For backward compatibility
        data = self.all_data_seq[key][time_idx]
        return data

    def get_data_seq(self, key):
        """Get data sequence."""
        key = DataKey.replace_deprecated_key(key)  # For backward compatibility
        data_seq = self.all_data_seq[key]
        return data_seq

    def get_meta_data(self, key):
        """Get meta data."""
        return self.meta_data[key]

    def calc_rel_data(self, key, all_data_seq=None):
        """Calculate relative data."""
        if all_data_seq is None:
            all_data_seq = self.all_data_seq

        abs_key = DataKey.get_abs_key(key)

        if key in (DataKey.MEASURED_JOINT_POS_REL, DataKey.COMMAND_JOINT_POS_REL):
            if len(all_data_seq[abs_key]) < 2:
                return np.zeros_like(all_data_seq[abs_key][0])
            else:
                current_pos = all_data_seq[abs_key][-1]
                prev_pos = all_data_seq[abs_key][-2]
                return current_pos - prev_pos
        elif key in (DataKey.MEASURED_EEF_POSE_REL, DataKey.COMMAND_EEF_POSE_REL):
            if len(all_data_seq[abs_key]) < 2:
                return np.zeros(6)
            else:
                current_se3 = get_se3_from_pose(all_data_seq[abs_key][-1])
                prev_se3 = get_se3_from_pose(all_data_seq[abs_key][-2])
                return get_rel_pose_from_se3(prev_se3.actInv(current_se3))

    def save_data(
        self, filename, all_data_seq=None, meta_data=None, increment_episode_idx=True
    ):
        """Save data."""
        if all_data_seq is None:
            all_data_seq = self.all_data_seq
        if meta_data is None:
            meta_data = self.meta_data

        # For backward compatibility
        for orig_key in all_data_seq.keys():
            new_key = DataKey.replace_deprecated_key(orig_key)
            if orig_key != new_key:
                all_data_seq[new_key] = all_data_seq.pop(orig_key)

        # Dump to a file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with h5py.File(filename, "w") as h5file:
            for key in all_data_seq.keys():
                if isinstance(all_data_seq[key], list):
                    h5file.create_dataset(key, data=np.array(all_data_seq[key]))
                else:
                    raise ValueError(
                        f"[{self.__class__.__name__}] Unsupported type of data sequence: {type(all_data_seq[key])}"
                    )
            for key in meta_data.keys():
                h5file.attrs[key] = meta_data[key]

        if increment_episode_idx:
            self.episode_idx += 1

    def load_data(self, filename, load_keys=None):
        """Load data."""
        self.all_data_seq = {}
        self.meta_data = {}
        with h5py.File(filename, "r") as h5file:
            for orig_key in h5file.keys():
                new_key = DataKey.replace_deprecated_key(
                    orig_key
                )  # For backward compatibility
                if (
                    load_keys is not None
                    and orig_key not in load_keys
                    and new_key not in load_keys
                ):
                    continue
                self.all_data_seq[new_key] = h5file[orig_key][:]
            for key in h5file.attrs.keys():
                self.meta_data[key] = h5file.attrs[key]

    def reverse_data(self, all_data_seq=None):
        """Reverse sequence data."""
        if all_data_seq is None:
            all_data_seq = self.all_data_seq

        for key in all_data_seq.keys():
            if key == DataKey.TIME:
                continue
            elif isinstance(all_data_seq[key], list):
                all_data_seq[key].reverse()
            else:
                raise ValueError(
                    f"[{self.__class__.__name__}] Unsupported type of data sequence: {type(all_data_seq[key])}"
                )

            if key in (
                DataKey.MEASURED_JOINT_VEL,
                DataKey.COMMAND_JOINT_VEL,
                DataKey.MEASURED_EEF_VEL,
                DataKey.COMMAND_EEF_VEL,
                DataKey.MEASURED_JOINT_POS_REL,
                DataKey.COMMAND_JOINT_POS_REL,
            ):
                all_data_seq[key] = [-1.0 * data for data in all_data_seq[key]]
            elif key in (DataKey.MEASURED_EEF_POSE_REL, DataKey.COMMAND_EEF_POSE_REL):
                all_data_seq[key] = [
                    get_rel_pose_from_se3(get_se3_from_rel_pose(data).inverse())
                    for data in all_data_seq[key]
                ]

    def setup_env_world(self, world_idx=None):
        """Setup the environment world."""
        if world_idx is None:
            kwargs = {"cumulative_idx": self.episode_idx}
        else:
            kwargs = {"world_idx": world_idx}
        self.world_idx = self.env.unwrapped.modify_world(**kwargs)
        self.meta_data["world_idx"] = self.world_idx

    def setup_camera_info(self):
        """Set camera info."""
        self.meta_data["camera_names"] = self.env.unwrapped.camera_names
        self.meta_data["tactile_names"] = self.env.unwrapped.tactile_names
        for camera_name in self.env.unwrapped.camera_names:
            depth_key = DataKey.get_depth_image_key(camera_name)
            self.meta_data[depth_key + "_fovy"] = self.env.unwrapped.get_camera_fovy(
                camera_name
            )
