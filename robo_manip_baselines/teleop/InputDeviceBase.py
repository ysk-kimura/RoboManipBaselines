from abc import ABC, abstractmethod


class InputDeviceBase(ABC):
    """Base class for teleoperation input device."""

    def __init__(self, motion_manager):
        self.motion_manager = motion_manager

        self.connected = False
        self.state = None

    @abstractmethod
    def connect(self):
        pass

    def is_ready(self):
        return True

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def set_command_data(self):
        pass
