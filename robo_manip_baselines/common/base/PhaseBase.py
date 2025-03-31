from abc import ABC, abstractmethod

from ..data.DataKey import DataKey


class PhaseBase(ABC):
    """
    Base class for phase of robot operation.

    The op (meaning "operation") in the member variables is an instance of TeleopBase or RolloutBase.
    """

    def __init__(self, op):
        self.op = op

    def start(self):
        self.start_time = self.op.env.unwrapped.get_time()

    def pre_update(self):
        """Pre-update, which is called before the environment step."""
        pass

    def post_update(self):
        """Post-update, which is called after the environment step."""
        pass

    def check_transition(self):
        return False  # Never transition from this phase

    def get_elapsed_duration(self):
        return self.op.env.unwrapped.get_time() - self.start_time

    @property
    def name(self):
        return self.__class__.__name__


class ReachPhaseBase(PhaseBase, ABC):
    def start(self):
        super().start()

        self.set_target()

    def pre_update(self):
        self.op.motion_manager.set_command_data(
            DataKey.COMMAND_EEF_POSE, self.target_se3
        )

    def check_transition(self):
        return self.get_elapsed_duration() > self.duration

    @abstractmethod
    def set_target(self):
        pass


class GraspPhaseBase(PhaseBase, ABC):
    def start(self):
        super().start()

        self.set_target()

    def pre_update(self):
        self.op.motion_manager.set_command_data(
            DataKey.COMMAND_GRIPPER_JOINT_POS, self.gripper_joint_pos
        )

    def check_transition(self):
        return self.get_elapsed_duration() > self.duration

    @abstractmethod
    def set_target(self):
        pass

    def set_target_close(self):
        self.gripper_joint_pos = self.op.env.action_space.high[
            self.op.env.unwrapped.gripper_joint_idxes
        ]
        self.duration = 0.5  # [s]

    def set_target_open(self):
        self.gripper_joint_pos = self.op.env.action_space.low[
            self.op.env.unwrapped.gripper_joint_idxes
        ]
        self.duration = 0.5  # [s]
