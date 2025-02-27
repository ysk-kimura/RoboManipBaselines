from tqdm import tqdm
import numpy as np
import glob
import zarr
import argparse
import os
import cv2
from multiprocessing import Pool
from robo_manip_baselines.common import DataKey, DataManager

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--skip", type=int, default=3)
parser.add_argument("--train_keywords", nargs="*", required=False)
parser.add_argument("-j", "--nproc", type=int, default=1)
args = parser.parse_args()

in_file_names = glob.glob(os.path.join(args.in_dir, "**/*.npz"), recursive=True)
if args.train_keywords is not None:
    in_file_names = [
        name
        for name in in_file_names
        if any([(word in name) for word in args.train_keywords])
    ]
in_file_names.sort()

actions = None
joints = None
images = None
ep_ends = np.zeros(len(in_file_names), dtype=np.int64)


def get_data(in_file_name):
    print(" " * 4 + f"{in_file_name}")
    data_manager = DataManager(env=None)
    data_manager.load_data(in_file_name)
    _actions = data_manager.get_data(DataKey.COMMAND_JOINT_POS)[:: args.skip]
    _joints = data_manager.get_data(DataKey.MEASURED_JOINT_POS)[:: args.skip]
    _images = data_manager.get_data(DataKey.get_rgb_image_key("front"))[:: args.skip]
    return (_actions, _joints, _images)


print("[convert_npz_to_zarr] Get npz files:")
pool = Pool(args.nproc)
data = pool.map(get_data, in_file_names)

print("[convert_npz_to_zarr] Concatenate:")
for idx, (_actions, _joints, _images) in enumerate(tqdm(data)):
    if idx == 0:
        actions = _actions
        joints = _joints
        images = _images
        ep_ends[idx] = len(_joints)
    else:
        actions = np.concatenate((actions, _actions))
        joints = np.concatenate((joints, _joints))
        images = np.concatenate((images, _images))
        ep_ends[idx] = ep_ends[idx - 1] + len(_joints)

# https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/config/task/real_pusht_image.yaml#L3
images = np.array([cv2.resize(image, (320, 240)) for image in images])

print(f"[convert_npz_to_zarr] Save a zarr file: {args.out_dir}")
zarr_root = zarr.open(args.out_dir, mode="w")
zarr_root.create_group("meta")
zarr_root["meta"].create_dataset("episode_ends", data=ep_ends)
zarr_root.create_group("data")
zarr_root["data"].create_dataset("action", data=actions)
zarr_root["data"].create_dataset("joint", data=joints)
zarr_root["data"].create_dataset("img", data=images)

for key in ("meta/episode_ends", "data/action", "data/joint", "data/img"):
    shape = zarr_root[f"{key}"].shape
    print(" " * 4 + f"{key}:\t{shape}")
print(" " * 4 + f"episode_ends: {list(zarr_root.meta.episode_ends)}")
