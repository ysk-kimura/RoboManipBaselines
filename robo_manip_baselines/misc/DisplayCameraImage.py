import argparse
import os
import re

import cv2


class DisplayCameraImage:
    def __init__(self, camera_device_name):
        self.camera_device_name = camera_device_name
        self.cap = self.find_device_capture(camera_device_name)

    def find_device_capture(self, camera_device_name):
        print(f"[{self.__class__.__name__}] {camera_device_name=}")
        cap = None
        for device_name in os.listdir("/sys/class/video4linux"):
            print(
                f"[{self.__class__.__name__}] {self.find_device_capture.__name__}, {device_name=}"
            )
            real_device_name = os.path.realpath(
                "/sys/class/video4linux/" + device_name + "/name"
            )
            with open(real_device_name, "r", encoding="utf-8") as device_name_file:
                detected_device_id = device_name_file.read().rstrip()
            if camera_device_name in detected_device_id:
                device_num = int(re.search(r"\d+$", device_name).group(0))
                print(
                    f"[{self.__class__.__name__}] {self.find_device_capture.__name__}, "
                    f"Found device. ID: {detected_device_id}"
                )
                cap = cv2.VideoCapture(device_num)
                if cap is None or not cap.isOpened():
                    print(
                        f"[{self.__class__.__name__}] {self.find_device_capture.__name__}, "
                        f"Unable to open video source ({device_num=})."
                    )
                    continue
                break
        if cap is None:
            raise LookupError()
        return cap

    def display_camera_output(
        self, horizontal_size, vertical_size, x_position, y_position
    ):
        cv2.namedWindow(f"{self.camera_device_name} Live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            f"{self.camera_device_name} Live", horizontal_size, vertical_size
        )
        cv2.moveWindow(f"{self.camera_device_name} Live", x_position, y_position)

        print(f"[{self.__class__.__name__}] Press q on image to exit.")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                resize_frame = cv2.resize(frame, (horizontal_size, vertical_size))
                cv2.imshow(f"{self.camera_device_name} Live", resize_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera_device_name", type=str, default="Webcam")
    parser.add_argument("-w", "--horizontal_size", default=810)
    parser.add_argument("-v", "--vertical_size", default=490)
    parser.add_argument("-x", "--x_position", default=0)
    parser.add_argument("-y", "--y_position", default=0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        disp = DisplayCameraImage(args.camera_device_name)
        disp.display_camera_output(
            args.horizontal_size, args.vertical_size, args.x_position, args.y_position
        )
    except LookupError:
        print(
            f"[{DisplayCameraImage.__class__.__name__}] Error: "
            f"No device matching '{args.camera_device_name}' was found."
        )


if __name__ == "__main__":
    main()
