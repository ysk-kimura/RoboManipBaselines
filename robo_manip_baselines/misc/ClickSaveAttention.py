#!/usr/bin/env python3

import argparse
import glob
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


class ClickSaveAttention:
    def __init__(self, dataset_dir, camera_name):
        self.dataset_dir = dataset_dir
        self.camera_name = camera_name

    def get_rmb_dirs(self):
        """Return subdirectories under dataset_dir, or glob results if it contains wildcards."""
        if any(ch in self.dataset_dir for ch in ("*", "?", "[")):
            dirs = sorted([d for d in glob.glob(self.dataset_dir) if os.path.isdir(d)])
        else:
            if os.path.isdir(self.dataset_dir):
                dirs = sorted(
                    [
                        os.path.join(self.dataset_dir, name)
                        for name in os.listdir(self.dataset_dir)
                        if os.path.isdir(os.path.join(self.dataset_dir, name))
                    ]
                )
            else:
                dirs = []
        return dirs

    def find_mp4_for_dir(self, d: str):
        """Recursively search under dataset_dir for a matching mp4 for camera_name."""
        pattern = os.path.join(
            self.dataset_dir, "**", f"{self.camera_name}_rgb_image.rmb.mp4"
        )
        candidates = glob.glob(pattern, recursive=True)
        if not candidates:
            return None

        basename = os.path.basename(os.path.normpath(d))
        preferred = [p for p in candidates if basename and (basename in p)]
        if preferred:
            return preferred[0]
        return candidates[0]

    def read_first_frame(self, mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def ask_click_on_image(self, image, title=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)
        if title:
            ax.set_title(title)
        ax.axis("off")

        pts = plt.ginput(1, timeout=0)  # wait for one click
        plt.close(fig)
        if not pts:
            return None
        return pts[0]  # (x, y) floats

    def create_broadcast_array(self, x_i, y_i, n):
        """Return an (n,2) int32 array filled with (x_i,y_i)."""
        if n <= 0:
            return np.zeros((0, 2), dtype=np.int32)
        pair = np.array([x_i, y_i], dtype=np.int32)
        return np.tile(pair.reshape(1, 2), (n, 1))

    def broadcast_click_to_all_hdf5(self, rmb_dirs, x_i, y_i):
        """For each directory in rmb_dirs, save an (n,2) dataset where n = len(time)."""
        ds_name = f"{self.camera_name}_attention"
        for d in rmb_dirs:
            h5_path = os.path.join(d, "main.rmb.hdf5")
            if not os.path.exists(h5_path):
                print(f"[{self.__class__.__name__}] SKIP {d} : {h5_path} not found.")
                continue

            try:
                with h5py.File(h5_path, "a") as f:
                    if "time" not in f:
                        print(
                            f"[{self.__class__.__name__}] SKIP {d} : 'time' dataset not found in {h5_path}."
                        )
                        continue
                    # determine n from time dataset's first dimension
                    time_ds = f["time"]
                    try:
                        n = int(time_ds.shape[0])
                    except Exception:
                        # fallback: use size
                        n = int(time_ds.size)

                    data = self.create_broadcast_array(x_i, y_i, n)
                    if ds_name in f:
                        del f[ds_name]
                    f.create_dataset(ds_name, data=data, dtype=np.int32)
                    print(
                        f"[{self.__class__.__name__}] SAVED {d} : {ds_name} shape={data.shape} -> {h5_path}"
                    )
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] ERROR {d} : Failed to write {ds_name} to {h5_path}: {e}"
                )

    def process_all_rmbs(self):
        rmb_dirs = self.get_rmb_dirs()
        if not rmb_dirs:
            print(
                f"[{self.__class__.__name__}] No target RMB directories found. dataset_dir: {self.dataset_dir}"
            )
            return

        # find first valid directory to obtain a click
        first_click_dir = None
        first_mp4 = None
        for d in rmb_dirs:
            mp4 = self.find_mp4_for_dir(d)
            h5 = os.path.join(d, "main.rmb.hdf5")
            if mp4 is None or not os.path.exists(mp4):
                continue
            if not os.path.exists(h5):
                continue
            img = self.read_first_frame(mp4)
            if img is None:
                continue
            # this directory is valid for obtaining the click
            first_click_dir = d
            first_mp4 = mp4
            break

        if first_click_dir is None:
            print(
                f"[{self.__class__.__name__}] ERROR : No valid directory found for clicking (no mp4/hdf5/readable frame)."
            )
            return

        # show image from first valid directory and get click
        title = f"{os.path.basename(first_click_dir)} - Click a point on the image (close window or Esc to skip)\n(mp4: {os.path.relpath(first_mp4, start=self.dataset_dir)})"
        img = self.read_first_frame(first_mp4)
        click = self.ask_click_on_image(img, title=title)
        if click is None:
            print(
                f"[{self.__class__.__name__}] INFO : Click cancelled. No changes made."
            )
            return

        x_f, y_f = click
        x_i, y_i = int(round(x_f)), int(round(y_f))
        print(f"[{self.__class__.__name__}] Clicked coordinates: ({x_i}, {y_i})")

        # Clip coordinates inside image
        h, w = img.shape[:2]
        x_i = max(0, min(w - 1, x_i))
        y_i = max(0, min(h - 1, y_i))

        # Broadcast and save to all hdf5 files
        self.broadcast_click_to_all_hdf5(rmb_dirs, x_i, y_i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="./dataset/", help="dataset directory"
    )
    parser.add_argument("--camera_name", type=str, default="front", help="camera name")
    args = parser.parse_args()

    clicker = ClickSaveAttention(
        dataset_dir=args.dataset_dir, camera_name=args.camera_name
    )
    clicker.process_all_rmbs()
