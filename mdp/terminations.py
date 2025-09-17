
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import axis_angle_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_fallen(
    env: ManagerBasedRLEnv,
    thresh_rad: float = 1.2,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    rpy = axis_angle_from_quat(robot.data.root_quat_w)
    rpy = torch.abs(rpy)
    fallen = (rpy[:, 0] > thresh_rad) | (rpy[:, 1] > thresh_rad)
    return fallen

