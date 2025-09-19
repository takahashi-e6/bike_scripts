# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from managed_bike.robots.bike import BIKE_CONFIG  # isort:skip

##
# Scene definition
##


@configclass
class ManagedBikeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = BIKE_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(.0, 1.0), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-math.pi/2, math.pi/2), heading=(-math.pi/4, math.pi/4)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    back_wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["back_wheel_joint"], scale=2*math.pi)
    steering_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["steering_joint"], scale=1.57)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        omg_r = ObsTerm(func=mdp.roll_omg)
        # roll_and_pitch = ObsTerm(func=mdp.roll_and_pitch)
        alignment = ObsTerm(func=mdp.alignment, params={"command_name": "base_velocity"})

        forward_speed = ObsTerm(func=mdp.forward_speed)


        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"roll": (-0.2, 0.2)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    omg_r_reward = RewTerm(
        func=mdp.omg_reward,
        weight=1.0,
    )
    forward = RewTerm(
        func=mdp.forward_reward,
        weight=2.0,
    )
    alignment = RewTerm(
        func=mdp.alignment_reward,
        weight=1.0,
        params={"command_name": "base_velocity"}
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Fall down
    fall_down = DoneTerm(
        func=mdp.is_fallen,
        params={"thresh_rad": 1.2,
                "robot_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class ManagedBikeEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ManagedBikeSceneCfg = ManagedBikeSceneCfg(num_envs=1000, env_spacing=4.0)
    # Basic settings
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
