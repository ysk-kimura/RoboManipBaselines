from os import path

import mujoco
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5ePickEnv(MujocoUR5eEnvBase):
    sim_timestep = 0.002
    frame_skip = 16
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(1 / (sim_timestep * frame_skip)),
    }

    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_pick.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.55 * np.pi,
                    -0.45 * np.pi,
                    np.pi / 2,
                    np.pi,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.obj_name_list = [f"obj{i}" for i in range(1, 10)]
        self.original_obj_pos_list = [
            self.model.body(obj_name).pos.copy() for obj_name in self.obj_name_list
        ]
        self.basket_name_list = [f"basket{i}" for i in range(1, 3)]
        self.original_basket_pos_list = [
            self.model.body(basket_name).pos.copy()
            for basket_name in self.basket_name_list
        ]
        self.basket_pos_offsets = np.array(
            [
                [-0.09, 0.0, 0.0],
                [-0.06, 0.0, 0.0],
                [-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.06, 0.0, 0.0],
            ]
        )  # [m]

    # def _get_success(self):
    #     pass

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.basket_pos_offsets)

        basket_pos_offset = self.basket_pos_offsets[world_idx]
        for basket_name, original_basket_pos in zip(
            self.basket_name_list, self.original_basket_pos_list
        ):
            self.model.body(basket_name).pos = original_basket_pos + basket_pos_offset

        if self.world_random_scale is not None:
            for obj_name, original_obj_pos in zip(
                self.obj_name_list, self.original_obj_pos_list
            ):
                obj_pos_offset = np.random.uniform(
                    low=-1.0 * self.world_random_scale,
                    high=self.world_random_scale,
                    size=3,
                )
                obj_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj_name}_freejoint"
                )
                obj_qpos_addr = self.model.jnt_qposadr[obj_joint_id]
                self.init_qpos[obj_qpos_addr : obj_qpos_addr + 3] = (
                    original_obj_pos + obj_pos_offset
                )

        return world_idx
