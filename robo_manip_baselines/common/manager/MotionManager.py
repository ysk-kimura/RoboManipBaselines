import numpy as np
import pinocchio as pin

from ..data.DataKey import DataKey
from ..utils.MathUtils import (
    get_pose_from_se3,
    get_se3_from_pose,
    get_se3_from_rel_pose,
)


class MotionManager:
    """
    Motion manager for robot arm and gripper.

    The manager computes forward and inverse kinematics using Pinocchio, a robot modeling library.
    """

    def __init__(self, env):
        self.env = env

        # Setup pinocchio model and data
        self.pin_model = pin.buildModelFromUrdf(self.env.unwrapped.arm_urdf_path)
        if self.env.unwrapped.arm_root_pose is not None:
            arm_root_se3 = get_se3_from_pose(self.env.unwrapped.arm_root_pose)
            self.pin_model.jointPlacements[1] = arm_root_se3.act(
                self.pin_model.jointPlacements[1]
            )
        self.pin_data = self.pin_model.createData()
        self.pin_data_obs = self.pin_model.createData()

        # Setup robot state
        self.joint_pos = np.concatenate(
            [
                self.env.unwrapped.init_qpos[self.env.unwrapped.arm_joint_idxes],
                self.env.unwrapped.init_qpos[self.env.unwrapped.gripper_joint_idxes],
            ]
        )
        self.forward_kinematics()
        self._original_target_se3 = self.pin_data.oMi[
            self.env.unwrapped.ik_eef_joint_id
        ].copy()
        self.target_se3 = self._original_target_se3.copy()

    def reset(self):
        """Reset states of arm and gripper."""
        # Reset robot state
        self.joint_pos = np.concatenate(
            [
                self.env.unwrapped.init_qpos[self.env.unwrapped.arm_joint_idxes],
                self.env.unwrapped.init_qpos[self.env.unwrapped.gripper_joint_idxes],
            ]
        )
        self.forward_kinematics()
        self.target_se3 = self._original_target_se3.copy()

    def forward_kinematics(self):
        """Solve forward kinematics."""
        pin.forwardKinematics(
            self.pin_model,
            self.pin_data,
            self.joint_pos[self.env.unwrapped.arm_joint_idxes],
        )

    def inverse_kinematics(self):
        """Solve inverse kinematics."""
        # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_d-inverse-kinematics.html
        error_se3 = self.current_se3.actInv(self.target_se3)
        error_vec = pin.log(error_se3).vector  # in joint frame
        arm_joint_pos = self.joint_pos[self.env.unwrapped.arm_joint_idxes]
        J = pin.computeJointJacobian(
            self.pin_model,
            self.pin_data,
            arm_joint_pos,
            self.env.unwrapped.ik_eef_joint_id,
        )  # in joint frame
        J = -1 * np.dot(pin.Jlog6(error_se3.inverse()), J)
        damping_scale = 1e-6
        delta_arm_joint_pos = -1 * J.T.dot(
            np.linalg.solve(
                J.dot(J.T)
                + (np.dot(error_vec, error_vec) + damping_scale) * np.identity(6),
                error_vec,
            )
        )
        self.joint_pos[self.env.unwrapped.arm_joint_idxes] = pin.integrate(
            self.pin_model, arm_joint_pos, delta_arm_joint_pos
        )
        self.forward_kinematics()

    @property
    def current_se3(self):
        """Get the current pose of the end-effector."""
        return self.pin_data.oMi[self.env.unwrapped.ik_eef_joint_id]

    def set_command_data(self, key, command, is_skip=False):
        """Set command data."""
        if key == DataKey.COMMAND_JOINT_POS:
            self.joint_pos[self.env.unwrapped.arm_joint_idxes] = command[
                self.env.unwrapped.arm_joint_idxes
            ]
            self.forward_kinematics()
            self.target_se3 = self.current_se3.copy()
            self.set_command_data(
                DataKey.COMMAND_GRIPPER_JOINT_POS,
                command[self.env.unwrapped.gripper_joint_idxes],
            )
        elif key == DataKey.COMMAND_JOINT_POS_REL:
            joint_pos = self.joint_pos.copy()
            if not is_skip:
                joint_pos += command
            self.set_command_data(DataKey.COMMAND_JOINT_POS, joint_pos)
        elif key == DataKey.COMMAND_GRIPPER_JOINT_POS:
            self.joint_pos[self.env.unwrapped.gripper_joint_idxes] = np.clip(
                command,
                self.env.action_space.low[self.env.unwrapped.gripper_joint_idxes],
                self.env.action_space.high[self.env.unwrapped.gripper_joint_idxes],
            )
        elif key == DataKey.COMMAND_EEF_POSE:
            if isinstance(command, pin.SE3):
                self.target_se3 = command
            else:
                self.target_se3 = get_se3_from_pose(command)
            self.inverse_kinematics()
        elif key == DataKey.COMMAND_EEF_POSE_REL:
            target_se3 = self.target_se3.copy()
            if not is_skip:
                target_se3 = target_se3 * get_se3_from_rel_pose(command)
            self.set_command_data(DataKey.COMMAND_EEF_POSE, target_se3)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def get_data(self, key, obs=None):
        """Get data of the specified key."""
        if key in DataKey.MEASURED_DATA_KEYS:
            return self.get_measured_data(key, obs)
        elif key in DataKey.COMMAND_DATA_KEYS:
            return self.get_command_data(key)
        else:
            raise ValueError(f"[{self.__class__.__name__}] Invalid data key: {key}")

    def get_measured_data(self, key, obs):
        """Get measured data of the specified key from observation."""
        if key == DataKey.MEASURED_JOINT_POS:
            return self.env.unwrapped.get_joint_pos_from_obs(obs)
        elif key == DataKey.MEASURED_JOINT_VEL:
            return self.env.unwrapped.get_joint_vel_from_obs(obs)
        elif key == DataKey.MEASURED_GRIPPER_JOINT_POS:
            return self.env.unwrapped.get_joint_pos_from_obs(obs)[
                self.env.unwrapped.gripper_joint_idxes
            ]
        elif key == DataKey.MEASURED_EEF_POSE:
            measured_arm_joint_pos = self.env.unwrapped.get_joint_pos_from_obs(obs)[
                self.env.unwrapped.arm_joint_idxes
            ]
            pin.forwardKinematics(
                self.pin_model, self.pin_data_obs, measured_arm_joint_pos
            )
            measured_se3 = self.pin_data_obs.oMi[self.env.unwrapped.ik_eef_joint_id]
            return get_pose_from_se3(measured_se3)
        elif key == DataKey.MEASURED_EEF_WRENCH:
            return self.env.unwrapped.get_eef_wrench_from_obs(obs)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid measured data key: {key}"
            )

    def get_command_data(self, key):
        """Get command data of the specified key."""
        if key == DataKey.COMMAND_JOINT_POS:
            return self.joint_pos.copy()
        elif key == DataKey.COMMAND_GRIPPER_JOINT_POS:
            return self.joint_pos[self.env.unwrapped.gripper_joint_idxes].copy()
        elif key == DataKey.COMMAND_EEF_POSE:
            return get_pose_from_se3(self.target_se3)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def draw_markers(self):
        """Draw markers of the current and target poses of the end-effector to viewer."""
        self.env.unwrapped.draw_box_marker(
            pos=self.target_se3.translation,
            mat=self.target_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(0, 1, 0, 0.5),
        )
        self.env.unwrapped.draw_box_marker(
            pos=self.current_se3.translation,
            mat=self.current_se3.rotation,
            size=(0.02, 0.02, 0.03),
            rgba=(1, 0, 0, 0.5),
        )
