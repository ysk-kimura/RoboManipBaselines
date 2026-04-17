import time

import numpy as np
import openvr

# あなたの環境に合わせて必要なら変更
DEVICE_MAP = {"LHR-55377A2B": "right_wrist"}


def get_gripper_axis_direction(axis_value):
    if axis_value > 0.0:
        return 1
    if axis_value < 0.0:
        return -1
    return 0


def main():
    print("Initializing OpenVR...")
    vr_system = openvr.init(openvr.VRApplication_Other)

    try:
        print("Start reading controller input (Ctrl+C to exit)")
        while True:
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding,
                0,
                openvr.k_unMaxTrackedDeviceCount,
            )

            for i in range(openvr.k_unMaxTrackedDeviceCount):
                pose = poses[i]

                if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
                    continue

                device_class = vr_system.getTrackedDeviceClass(i)
                if device_class != openvr.TrackedDeviceClass_Controller:
                    continue

                sn = vr_system.getStringTrackedDeviceProperty(
                    i, openvr.Prop_SerialNumber_String
                )

                if sn not in DEVICE_MAP:
                    continue

                name = DEVICE_MAP[sn]

                _, controller_state = vr_system.getControllerState(i)

                axes = np.array(
                    [
                        controller_state.rAxis[0].x,
                        controller_state.rAxis[0].y,
                        controller_state.rAxis[1].x,  # ← これがトリガー
                    ]
                )

                gripper_dir = get_gripper_axis_direction(axes[1])

                print(f"[{name}] axes={axes} | gripper_dir={gripper_dir}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
