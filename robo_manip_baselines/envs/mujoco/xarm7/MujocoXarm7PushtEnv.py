from os import path

import mujoco
import numpy as np

from .MujocoXarm7EnvBase import MujocoXarm7EnvBase


class MujocoXarm7PushtEnv(MujocoXarm7EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoXarm7EnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/xarm7/env_xarm7_pusht.xml",
            ),
            np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0, *[0.0] * 6]),
            **kwargs,
        )

        self.original_tblock_pos = self.model.body("tblock").pos.copy()
        self.tblock_pos_offsets = np.array(
            [
                [0.0, -0.06, 0.0],
                [0.0, -0.03, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.06, 0.0],
                [0.0, 0.09, 0.0],
            ]
        )  # [m]

    def _get_reward(self):
        tblock_pos = self.data.body("tblock").xpos.copy()
        tblock_angle = self.data.body("tblock").xquat.copy()
        target_pos = self.data.body("target_region").xpos.copy()
        target_angle = self.data.body("target_region").xquat.copy()

        xy_thre = 0.03  # [m]
        if np.max(np.abs(tblock_pos[:2] - target_pos[:2])) < xy_thre:
            xy_success = True
        else:
            xy_success = False
        rot_thre = 0.02
        if np.abs(tblock_angle[0] - target_angle[0]) < rot_thre:
            rot_success = True
        else:
            rot_success = False

        return 1.0 if xy_success and rot_success else 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.tblock_pos_offsets)

        tblock_pos = self.original_tblock_pos + self.tblock_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            tblock_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        tblock_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "tblock_freejoint"
        )
        tblock_qpos_addr = self.model.jnt_qposadr[tblock_joint_id]
        self.init_qpos[tblock_qpos_addr : tblock_qpos_addr + 3] = tblock_pos

        return world_idx
