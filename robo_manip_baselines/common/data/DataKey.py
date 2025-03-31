import warnings

import numpy as np


def _calc_dim_from_idxes(idxes):
    if isinstance(idxes, list) or isinstance(idxes, np.ndarray):
        return len(idxes)
    elif isinstance(idxes, slice):
        start = idxes.start if idxes.start is not None else 0
        stop = idxes.stop
        step = idxes.step if idxes.step is not None else 1
        if stop is None:
            raise ValueError(f"[DataKey] The stop of slice must be specified: {idxes}")
        if step > 0:
            return max(0, (stop - start + step - 1) // step)
        else:
            return max(0, (start - stop - step - 1) // -step)
    else:
        raise ValueError(f"[DataKey] Unsupported type of idxes: {type(idxes)}")


class DataKey:
    """Data key."""

    # Time [s]
    TIME = "time"

    # Measured joint position (including both arm and gripper)
    MEASURED_JOINT_POS = "measured_joint_pos"
    # Command joint position (including both arm and gripper)
    COMMAND_JOINT_POS = "command_joint_pos"

    # Measured joint position relative to previous step (including both arm and gripper)
    MEASURED_JOINT_POS_REL = "measured_joint_pos_rel"
    # Command joint position relative to previous step (including both arm and gripper)
    COMMAND_JOINT_POS_REL = "command_joint_pos_rel"

    # Measured joint velocity
    MEASURED_JOINT_VEL = "measured_joint_vel"
    # Command joint velocity
    COMMAND_JOINT_VEL = "command_joint_vel"

    # Measured joint torque
    MEASURED_JOINT_TORQUE = "measured_joint_torque"
    # Command joint torque
    COMMAND_JOINT_TORQUE = "command_joint_torque"

    # Measured gripper joint position
    MEASURED_GRIPPER_JOINT_POS = "measured_gripper_joint_pos"
    # Command gripper joint position
    COMMAND_GRIPPER_JOINT_POS = "command_gripper_joint_pos"

    # Measured end-effector pose (tx, ty, tz, qw, qx, qy, qz)
    # Note: This is the end-effector pose corresponding to the measured joint position.
    MEASURED_EEF_POSE = "measured_eef_pose"
    # Command end-effector pose (tx, ty, tz, qw, qx, qy, qz)
    # Note: This is the target end-effector pose for IK, not the end-effector pose corresponding to the command joint position.
    COMMAND_EEF_POSE = "command_eef_pose"

    # Measured end-effector pose relative to previous step in the previous pose frame (tx, ty, tz, roll, pitch, yaw)
    MEASURED_EEF_POSE_REL = "measured_eef_pose_rel"
    # Command end-effector pose relative to previous step in the previous pose frame (tx, ty, tz, roll, pitch, yaw)
    COMMAND_EEF_POSE_REL = "command_eef_pose_rel"

    # Measured end-effector velocity (vx, vy, vz, wx, wy, wz)
    MEASURED_EEF_VEL = "measured_eef_vel"
    # Command end-effector velocity (vx, vy, vz, wx, wy, wz)
    COMMAND_EEF_VEL = "command_eef_vel"

    # Measured end-effector wrench (fx, fy, fz, nx, ny, nz)
    MEASURED_EEF_WRENCH = "measured_eef_wrench"
    # Command end-effector wrench (fx, fy, fz, nx, ny, nz)
    COMMAND_EEF_WRENCH = "command_eef_wrench"

    # All keys of measured data
    MEASURED_DATA_KEYS = [
        MEASURED_JOINT_POS,
        MEASURED_JOINT_POS_REL,
        MEASURED_JOINT_VEL,
        # MEASURED_JOINT_TORQUE,
        MEASURED_GRIPPER_JOINT_POS,
        MEASURED_EEF_POSE,
        MEASURED_EEF_POSE_REL,
        # MEASURED_EEF_VEL,
        MEASURED_EEF_WRENCH,
    ]

    # All keys of command data
    COMMAND_DATA_KEYS = [
        COMMAND_JOINT_POS,
        COMMAND_JOINT_POS_REL,
        # COMMAND_JOINT_VEL,
        # COMMAND_JOINT_TORQUE,
        COMMAND_GRIPPER_JOINT_POS,
        COMMAND_EEF_POSE,
        COMMAND_EEF_POSE_REL,
        # COMMAND_EEF_VEL,
        # COMMAND_EEF_WRENCH,
    ]

    @classmethod
    def get_dim(cls, key, env):
        """Get the dimension of the data specified by key."""
        if key == DataKey.TIME:
            return 1
        elif key in (
            DataKey.MEASURED_JOINT_POS,
            DataKey.COMMAND_JOINT_POS,
            DataKey.MEASURED_JOINT_POS_REL,
            DataKey.COMMAND_JOINT_POS_REL,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.COMMAND_JOINT_VEL,
            DataKey.MEASURED_JOINT_TORQUE,
            DataKey.COMMAND_JOINT_TORQUE,
        ):
            return _calc_dim_from_idxes(
                env.unwrapped.arm_joint_idxes
            ) + _calc_dim_from_idxes(env.unwrapped.gripper_joint_idxes)
        elif key in (
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
        ):
            return _calc_dim_from_idxes(env.unwrapped.gripper_joint_idxes)
        elif key in (DataKey.MEASURED_EEF_POSE, DataKey.COMMAND_EEF_POSE):
            return 7
        elif key in (DataKey.MEASURED_EEF_POSE_REL, DataKey.COMMAND_EEF_POSE_REL):
            return 6
        elif key in (DataKey.MEASURED_EEF_VEL, DataKey.COMMAND_EEF_VEL):
            return 6
        elif key in (DataKey.MEASURED_EEF_WRENCH, DataKey.COMMAND_EEF_WRENCH):
            return 6

    @classmethod
    def get_measured_key(cls, key):
        """Get measured key."""
        if key.startswith("measured_"):
            return key
        elif key.startswith("command_"):
            return "measured_" + key[len("command_") :]
        else:
            raise ValueError(f"[{cls.__name__}] Invalid data key: {key}")

    @classmethod
    def get_command_key(cls, key):
        """Get command key."""
        if key.startswith("command_"):
            return key
        elif key.startswith("measured_"):
            return "command_" + key[len("measured_") :]
        else:
            raise ValueError(f"[{cls.__name__}] Invalid data key: {key}")

    @classmethod
    def get_rel_key(cls, key):
        """Get the relative key corresponding to the absolute key."""
        if key == DataKey.MEASURED_JOINT_POS:
            return DataKey.MEASURED_JOINT_POS_REL
        elif key == DataKey.COMMAND_JOINT_POS:
            return DataKey.COMMAND_JOINT_POS_REL
        elif key == DataKey.MEASURED_EEF_POSE:
            return DataKey.MEASURED_EEF_POSE_REL
        elif key == DataKey.COMMAND_EEF_POSE:
            return DataKey.COMMAND_EEF_POSE_REL
        else:
            raise ValueError(f"[{cls.__name__}] Relative data key not found: {key}")

    @classmethod
    def get_abs_key(cls, key):
        """Get the absolute key corresponding to the relative key."""
        if key == DataKey.MEASURED_JOINT_POS_REL:
            return DataKey.MEASURED_JOINT_POS
        elif key == DataKey.COMMAND_JOINT_POS_REL:
            return DataKey.COMMAND_JOINT_POS
        elif key == DataKey.MEASURED_EEF_POSE_REL:
            return DataKey.MEASURED_EEF_POSE
        elif key == DataKey.COMMAND_EEF_POSE_REL:
            return DataKey.COMMAND_EEF_POSE
        else:
            raise ValueError(f"[{cls.__name__}] Absolute data key not found: {key}")

    @classmethod
    def get_rgb_image_key(cls, camera_name):
        """Get the rgb image key from the camera name."""
        return camera_name.lower() + "_rgb_image"

    @classmethod
    def get_depth_image_key(cls, camera_name):
        """Get the depth image key from the camera name."""
        return camera_name.lower() + "_depth_image"

    @classmethod
    def get_camera_name(cls, key):
        if cls.is_rgb_image_key(key):
            return key[: -len("_rgb_image")]
        elif cls.is_depth_image_key(key):
            return key[: -len("_depth_image")]
        else:
            raise ValueError(f"[{cls.__name__}] Not rgb or depth key: {key}")

    @classmethod
    def is_rgb_image_key(cls, key):
        """Check if the key is for RGB image."""
        return key.endswith("_rgb_image")

    @classmethod
    def is_depth_image_key(cls, key):
        """Check if the key is for depth image."""
        return key.endswith("_depth_image")

    @classmethod
    def get_plot_scale(cls, key, env):
        """Get scale to plot data."""
        if key in (
            DataKey.MEASURED_JOINT_POS,
            DataKey.COMMAND_JOINT_POS,
            DataKey.MEASURED_JOINT_POS_REL,
            DataKey.COMMAND_JOINT_POS_REL,
        ):
            return np.concatenate(
                [
                    np.ones(_calc_dim_from_idxes(env.unwrapped.arm_joint_idxes)),
                    np.full(
                        _calc_dim_from_idxes(env.unwrapped.gripper_joint_idxes), 0.01
                    ),
                ]
            )
        elif key in (
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
        ):
            return np.full(
                _calc_dim_from_idxes(env.unwrapped.gripper_joint_idxes), 0.01
            )
        elif key in (DataKey.MEASURED_EEF_POSE_REL, DataKey.COMMAND_EEF_POSE_REL):
            return np.full(6, 100.0)
        else:
            return np.ones(cls.get_dim(key, env))

    @classmethod
    def replace_deprecated_key(cls, orig_key):
        """Replace a deprecated key with a new key for backward compatibility."""
        if orig_key == "joint_pos":
            new_key = DataKey.MEASURED_JOINT_POS
        elif orig_key == "joint_vel":
            new_key = DataKey.MEASURED_JOINT_VEL
        elif orig_key == "wrench":
            new_key = DataKey.MEASURED_EEF_WRENCH
        elif orig_key == "measured_eef":
            new_key = DataKey.MEASURED_EEF_POSE
        elif orig_key == "command_eef":
            new_key = DataKey.COMMAND_EEF_POSE
        elif orig_key == "measured_wrench":
            new_key = DataKey.MEASURED_EEF_WRENCH
        elif orig_key == "command_wrench":
            new_key = DataKey.COMMAND_EEF_WRENCH
        elif orig_key == "action":
            new_key = DataKey.COMMAND_JOINT_POS
        else:
            new_key = orig_key.lower()
        if orig_key != new_key:
            warnings.warn(
                f'[{cls.__name__}] "{orig_key}" is deprecated, use "{new_key}" instead.'
            )
        return new_key
