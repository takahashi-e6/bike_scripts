
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def roll_and_pitch(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     robot: RigidObject = env.scene[robot_cfg.name]
#     euler = euler_xyz_from_quat(robot.data.root_link_quat_w)
#     roll = euler[:,0].reshape(-1,1)
#     pitch = euler[:,1].reshape(-1,1)
#     return torch.hstack((roll, pitch))

def roll_omg(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    return robot.data.root_com_ang_vel_b[:,0].reshape(-1,1)

# def pitch_omg(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     robot: RigidObject = env.scene[robot_cfg.name]
#     return robot.data.root_com_ang_vel_b[:,1].reshape(-1,1)

def calc(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity"
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    commands = env.command_manager.get_command(command_name)
    # ワールド座標系から見た速度ベクトルに変換
    forwards = quat_apply(robot.data.root_link_quat_w, robot.data.FORWARD_VEC_B)

    # commandsとforwardsの内積(向きの一致度)
    dot = torch.sum(forwards * commands, dim=-1, keepdim=True)
    # 外積で向きのずれの方向を計算
    cross = torch.cross(forwards, commands, dim=-1)[:,-1].reshape(-1,1)
    forward_speed = robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
    obs = torch.hstack((forward_speed, dot, cross))

    return obs