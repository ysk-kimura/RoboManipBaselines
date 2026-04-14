import time

import numpy as np
import openvr

openvr.init(openvr.VRApplication_Other)
vr = openvr.VRSystem()


def get_pose_matrix(pose):
    m = pose.mDeviceToAbsoluteTracking
    mat = np.array(
        [
            [m[0][0], m[0][1], m[0][2], m[0][3]],
            [m[1][0], m[1][1], m[1][2], m[1][3]],
            [m[2][0], m[2][1], m[2][2], m[2][3]],
            [0, 0, 0, 1],
        ]
    )
    return mat


try:
    while True:
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if not poses[i].bDeviceIsConnected:
                continue

            device_class = vr.getTrackedDeviceClass(i)

            if device_class == openvr.TrackedDeviceClass_Controller:
                pose = poses[i]

                if pose.bPoseIsValid:
                    mat = get_pose_matrix(pose)
                    pos = mat[:3, 3]
                    print(f"Controller {i} position:", pos)

                state = vr.getControllerState(i)[1]
                print("Trigger:", state.rAxis[1].x)
                print("Grip:", bool(state.ulButtonPressed >> openvr.k_EButton_Grip & 1))

        time.sleep(0.03)

finally:
    openvr.shutdown()
