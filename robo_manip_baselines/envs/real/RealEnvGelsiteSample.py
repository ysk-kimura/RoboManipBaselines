import os
import re

import cv2


class RealEnvGelsiteSample:
    def __init__(
        self,
    ):
        pass

    def setup_gelsight(self, tactile_ids):
        self.tactiles = {}

        for tactile_name, tactile_id in tactile_ids.items():
            for device_name in os.listdir("/sys/class/video4linux"):
                real_device_name = os.path.realpath(
                    "/sys/class/video4linux/" + device_name + "/name"
                )
                with open(real_device_name, "rt") as device_name_file:
                    detected_tactile_id = device_name_file.read().rstrip()
                is_found = tactile_id in detected_tactile_id
                print(
                    "{} {} -> {}".format(
                        "FOUND!" if is_found else "      ",
                        device_name,
                        detected_tactile_id,
                    )
                )
                if not is_found:
                    continue
                tactile_num = int(re.search("\d+$", device_name).group(0))
                tactile = cv2.VideoCapture(tactile_num)
                if tactile is None or not tactile.isOpened():
                    print(f"Warning: unable to open video source: {tactile_num}")
                    continue
                self.tactiles[tactile_name] = tactile
                break

    def _get_info(self):
        info = {}

        if len(self.tactile_names) == 0:
            return info

        # Get images
        info["rgb_images"] = {}
        info["depth_images"] = {}
        for tactile_name, tactile in self.tactiles.items():
            ret, rgb_image = tactile.read()
            assert ret, "ERROR! reading image from tactile!"
            info["rgb_images"][tactile_name] = rgb_image
            info["depth_images"][tactile_name] = None
            continue

        return info

    @property
    def tactile_names(self):
        return self.tactiles.keys()
