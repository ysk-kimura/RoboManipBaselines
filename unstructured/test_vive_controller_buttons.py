#!/usr/bin/env python3

import time

import openvr

# 対応付け
DEVICE_MAP = {
    "LHR-55377A2B": "right_wrist",
    "LHR-301CBF17": "left_wrist",
}


def main():
    vr_system = openvr.init(openvr.VRApplication_Other)

    try:
        while True:
            print("---- controller buttons ----")

            for device_idx in range(openvr.k_unMaxTrackedDeviceCount):
                device_class = vr_system.getTrackedDeviceClass(device_idx)
                if device_class != openvr.TrackedDeviceClass_Controller:
                    continue

                device_sn = vr_system.getStringTrackedDeviceProperty(
                    device_idx, openvr.Prop_SerialNumber_String
                )
                if device_sn not in DEVICE_MAP:
                    continue

                _, device_state = vr_system.getControllerState(device_idx)
                buttons = {
                    "application_menu": bool(
                        device_state.ulButtonPressed >> openvr.k_EButton_ApplicationMenu
                        & 1
                    ),
                    "grip": bool(
                        device_state.ulButtonPressed >> openvr.k_EButton_Grip & 1
                    ),
                    "trackpad_pressed": bool(
                        device_state.ulButtonPressed
                        >> openvr.k_EButton_SteamVR_Touchpad
                        & 1
                    ),
                    "trackpad_touched": bool(
                        device_state.ulButtonTouched
                        >> openvr.k_EButton_SteamVR_Touchpad
                        & 1
                    ),
                }

                print(f"[{DEVICE_MAP[device_sn]}] {buttons}")

            print("")
            time.sleep(1 / 30)

    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
