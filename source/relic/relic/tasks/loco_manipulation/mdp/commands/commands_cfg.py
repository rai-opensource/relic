# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from .arm_command import (
    ArmJointTrajectoryCommand,
    LegJointTrajectoryCommand,
    MultiLegJointTrajectoryCommand,
    BasePoseCommand,
    ArmLegJointBasePoseCommand,
)


@configclass
class ArmJointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = ArmJointTrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = MISSING
    """Length of the trajectory in seconds."""

    hold_time: tuple[float, float] = MISSING
    """Length of the arm holding in positon in seconds."""

    joint_names: tuple[str, ...] = MISSING

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gripper commands."""

        init_range: float = MISSING
        final_range: float = MISSING
        noise_range: float = MISSING

    ranges: Ranges = MISSING


@configclass
class LegJointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = LegJointTrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = MISSING
    """Length of the trajectory in seconds."""

    hold_time: tuple[float, float] = MISSING
    """Length of the arm holding in positon in seconds."""

    leg_joint_names: tuple[str, ...] = MISSING

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gripper commands."""

        init_range: float = MISSING
        final_range: float = MISSING
        noise_range: float = MISSING

    ranges: Ranges = MISSING


@configclass
class MultiLegJointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = MultiLegJointTrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = MISSING
    """Length of the trajectory in seconds."""

    hold_time: tuple[float, float] = MISSING
    """Length of the arm holding in positon in seconds."""

    leg_joint_names: dict = MISSING

    no_command_leg_prob: float = MISSING
    """Probability of not command any leg."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gripper commands."""

        init_range: float = MISSING
        final_range: float = MISSING
        noise_range: float = MISSING

    ranges: Ranges = MISSING


@configclass
class BasePoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = BasePoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gripper commands."""

        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        height: tuple[float, float] = MISSING
        noise_range: float = MISSING

    ranges: Ranges = MISSING


@configclass
class ArmLegJointBasePoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = ArmLegJointBasePoseCommand

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = (1.0, 3.0)
    """Length of the trajectory in seconds."""

    hold_time: tuple[float, float] = (0.5, 2.0)
    """Length of the arm holding in positon in seconds."""

    arm_joint_names: tuple[str, ...] = MISSING
    leg_joint_names: dict = MISSING

    command_range_roll: tuple[float, float] = (-0.35, 0.35)
    command_range_pitch: tuple[float, float] = (-0.35, 0.35)
    command_range_height: tuple[float, float] = (0.25, 0.65)
    command_range_arm_joint: tuple[list[float, ...], list[float, ...]] = (
        [-2.61799, -3.14159, 0.0, -2.79252, -1.8326, -2.87988, -1.5708],
        [3.14157, 0.52359, 3.14158, 2.79252, 1.83259, 2.87979, 0.0],
    )
    command_range_leg_joint: tuple[list[float, ...], list[float, ...]] = (
        [-0.7854, -0.89884, -2.7929],
        [0.78539, 2.29511, 0.0],
    )
    """Command sample ranges."""

    command_which_leg: int = 4
    """Which leg to command: -1: no leg; [0, 1, 2, 3]: [FL, FR, HL, HR]; 4: all leg"""
