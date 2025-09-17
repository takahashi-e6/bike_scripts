# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint position deviation from a target value."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # wrap the joint positions to (-pi, pi)
#     joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
#     # compute the reward
#     return torch.sum(torch.square(joint_pos - target), dim=1)

def omg_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing angular velocity around x axis (roll)"""
    asset: Articulation = env.scene[asset_cfg.name]
    omg_r = 2 - torch.square(asset.data.root_com_ang_vel_b[:,0])
    ret = torch.clamp(omg_r, min=0.0, max=2.0).reshape(-1,1)
    return ret.squeeze()

def forward_reward(env: ManagerBasedRLEnv,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    ) -> torch.Tensor:
    """Reward for forward velocity"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_com_lin_vel_b[:,0].squeeze()

def alignment_reward(env: ManagerBasedRLEnv,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    command_name: str = "base_velocity") -> torch.Tensor:
    """Reward for forward velocity"""
    asset: Articulation = env.scene[asset_cfg.name]

    # ロボット座標系でのxdot
    speed_vec = asset.data.root_com_lin_vel_b
    commands = env.command_manager.get_command(command_name)
    alignment_reward = torch.sum(speed_vec * commands, dim=-1, keepdim=True)

    # print("alignment_reward:", alignment_reward)
    return torch.exp(alignment_reward).squeeze()