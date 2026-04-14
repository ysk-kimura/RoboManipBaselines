import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase

# 対応付け　 "LHR-301CBF17": "left_wrist"
DEVICE_MAP = {"LHR-55377A2B": "right_wrist"}


class ViveControllerInputDevice(InputDeviceBase):
    """Vive Controller for teleoperation input device."""

    def __init__(
        self,
        arm_manager,
        pos_scale=1.0,
        gripper_scale=5.0,
        device_params={},
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.pos_scale = pos_scale
        self.gripper_scale = gripper_scale
        self.device_params = device_params

    def connect(self):
        self.enabled_teleop = False
        self.prev_enable_teleop = False
        self.controller_se3_at_enable = None
        self.eef_se3_at_enable = None
        if self.connected:
            return

        import openvr

        self.vr_system = openvr.init(openvr.VRApplication_Other)
        self.openvr = openvr
        self.connected = True

    def close(self):
        if self.connected:
            self.openvr.shutdown()
            self.connected = False

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        self._read_controller()
        if "right_wrist" not in self.state:
            self.prev_enable_teleop = False
            return

        enable_teleop = self.state["right_wrist"]["axes"][2] == 1.0
        if enable_teleop and not self.prev_enable_teleop:
            self.enabled_teleop = True
            self.controller_se3_at_enable = self.state["right_wrist"]["se3"].copy()
            self.eef_se3_at_enable = self.arm_manager.current_se3.copy()
            print(f"[{self.__class__.__name__}] Enable teleop.")
        self.prev_enable_teleop = enable_teleop

    def _read_controller(self):
        openvr = self.openvr
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )

        state = {}

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            pose = poses[i]

            if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
                continue

            device_class = self.vr_system.getTrackedDeviceClass(i)
            if device_class != openvr.TrackedDeviceClass_Controller:
                continue

            sn = self.vr_system.getStringTrackedDeviceProperty(
                i, openvr.Prop_SerialNumber_String
            )

            if sn not in DEVICE_MAP:
                continue

            name = DEVICE_MAP[sn]

            # 4x4行列
            mat = np.eye(4)
            mat[:3, :4] = pose.mDeviceToAbsoluteTracking.m

            # 位置・姿勢
            position = mat[:3, 3]
            se3 = pin.SE3(mat[:3, :3], position)

            _, controller_state = self.vr_system.getControllerState(i)
            axes = np.array(
                [
                    controller_state.rAxis[0].x,
                    controller_state.rAxis[0].y,
                    controller_state.rAxis[1].x,
                ]
            )
            buttons = {
                "application_menu": bool(
                    controller_state.ulButtonPressed >> openvr.k_EButton_ApplicationMenu
                    & 1
                ),
                "trackpad_pressed": bool(
                    controller_state.ulButtonPressed
                    >> openvr.k_EButton_SteamVR_Touchpad
                    & 1
                ),
            }

            # stateに格納
            state[name] = {
                "position": position,
                "se3": se3,
                "axes": axes,
                "buttons": buttons,
            }

        # 最後にまとめて代入（Spacemouseと同じ思想）
        self.state = state
        if "right_wrist" in state:
            # print(f"state: {self.state}")
            # print(f"position: {state['right_wrist']['position']}")
            # print(f"quat: {state['right_wrist']['quat']}")
            print(f"axes: {state['right_wrist']['axes']}")

    def is_ready(self):
        return (self.state is not None) and ("right_wrist" in self.state)

    def set_command_data(self):
        if (not self.enabled_teleop) or ("right_wrist" not in self.state):
            return

        # Set arm command
        controller_se3_rel = (
            self.controller_se3_at_enable.inverse() * self.state["right_wrist"]["se3"]
        )
        delta_pos_local = self.pos_scale * np.array(
            [
                -1.0 * controller_se3_rel.translation[1],
                1.0 * controller_se3_rel.translation[0],
                1.0 * controller_se3_rel.translation[2],
            ]
        )
        delta_pos_world = self.eef_se3_at_enable.rotation @ delta_pos_local
        axis_map = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        delta_rot = axis_map @ controller_se3_rel.rotation @ axis_map.T

        target_se3 = self.eef_se3_at_enable.copy()
        target_se3.translation += delta_pos_world
        target_se3.rotation = self.eef_se3_at_enable.rotation @ delta_rot

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
        buttons = self.state["right_wrist"]["buttons"]
        if buttons["application_menu"] and not buttons["trackpad_pressed"]:
            gripper_joint_pos -= self.gripper_scale
        elif buttons["trackpad_pressed"] and not buttons["application_menu"]:
            gripper_joint_pos += self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
