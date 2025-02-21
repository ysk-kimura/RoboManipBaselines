from os import path

import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eDoorEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_door.xml",
            ),
            np.array(
                [
                    np.pi,
                    -0.4 * np.pi,
                    -0.65 * np.pi,
                    -0.25 * np.pi,
                    np.pi / 2,
                    np.pi / 2,
                    0.0,
                ]
            ),
            **kwargs,
        )

        self.original_door_pos = self.model.body("door").pos.copy()
        self.door_pos_offsets = np.array(
            [
                [0.0, -0.06, 0.0],
                [0.0, -0.03, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.06, 0.0],
                [0.0, 0.09, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.door_pos_offsets)

        door_pos = self.original_door_pos + self.door_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            door_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.model.body("door").pos = door_pos

        return world_idx
