# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class UniformArmJointCommand(CommandTerm):
    """Command generator for generating arm joint commands uniformly."""

    cfg: CommandTermCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
        ----
            cfg: The configuration parameters for the command generator.
            env: The environment object.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._arm_joint_ids, self._arm_joint_names = self.robot.find_joints(
            self.cfg.arm_joint_names
        )

        self.lower_bound = self.robot.data.soft_joint_pos_limits[
            :, self._arm_joint_ids, 0
        ]
        self.upper_bound = self.robot.data.soft_joint_pos_limits[
            :, self._arm_joint_ids, 1
        ]

        # create buffers
        self.arm_command = torch.zeros(self.num_envs, 7, device=self.device)

        # -- metrics
        self.metrics["error_arm_pos"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformArmJointCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired arm command. Shape is (num_envs, 7)."""
        return self.arm_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # logs data
        self.metrics["error_arm_pos"] += (
            torch.norm(
                self.arm_command - self.robot.data.joint_pos[:, self._arm_joint_ids],
                dim=-1,
            )
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new arm joint targets
        self.arm_command[env_ids, :] = math_utils.sample_uniform(
            self.lower_bound[env_ids],
            self.upper_bound[env_ids],
            (len(env_ids), len(self._arm_joint_ids)),
            self.device,
        )

    def _update_command(self):
        pass


class ArmJointTrajectoryCommand(CommandTerm):
    """Command term that generates arm joint trajectory for spot arm."""

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
        ----
            cfg: The configuration parameters for the command term.
            env: The environment object.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.arm_joint_idxs = self.robot.find_joints(self.cfg.joint_names)[0]

        # create buffers to store the command
        # -- arm trajectory command
        self.arm_joint_start = torch.zeros(
            self.num_envs, len(self.arm_joint_idxs), device=self.device
        )
        self.arm_joint_goal = torch.zeros(
            self.num_envs, len(self.arm_joint_idxs), device=self.device
        )
        self.arm_joint_sub_goal = torch.zeros(
            self.num_envs, len(self.arm_joint_idxs), device=self.device
        )

        lower_bound = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_idxs, 0]
        upper_bound = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_idxs, 1]
        center = 0.5 * (lower_bound + upper_bound)
        half_range = 0.5 * (upper_bound - lower_bound)
        self.lower_bound = center - self.cfg.ranges.init_range * half_range
        self.upper_bound = center + self.cfg.ranges.init_range * half_range

        self.step_dt = env.step_dt
        self.timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()

        # -- metrics
        self.metrics["arm_joint_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "ArmJointTrajectoryCommandCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired arm joints in the environment frame. Shape is (num_envs, 7)."""
        return self.arm_joint_sub_goal

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the arm joint tracking error
        self.metrics["arm_joint_error"] = torch.norm(
            self.robot.data.joint_pos[:, self.arm_joint_idxs] - self.arm_joint_sub_goal,
            dim=1,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        self.arm_joint_start[env_ids] = torch.clamp(
            self.robot.data.joint_pos[env_ids][:, self.arm_joint_idxs],
            self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_idxs, 0],
            self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_idxs, 1],
        )
        self.arm_joint_sub_goal[env_ids] = self.arm_joint_start[env_ids]
        self.arm_joint_goal[env_ids] = math_utils.sample_uniform(
            self.lower_bound[env_ids],
            self.upper_bound[env_ids],
            (len(env_ids), len(self.arm_joint_idxs)),
            self.device,
        )

        self.traj_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.timer[env_ids] *= 0.0

    def _update_command(self):
        # step the timer
        self.timer += 1.0

        # update the mid_goal as timer goes
        reaching = self.timer <= self.traj_timesteps
        holding = torch.logical_and(
            self.traj_timesteps < self.timer,
            self.timer <= self.traj_timesteps + self.hold_timesteps,
        )
        reset = self.timer > self.traj_timesteps + self.hold_timesteps

        reaching_ids = reaching.nonzero(as_tuple=False).squeeze(-1)
        holding_ids = holding.nonzero(as_tuple=False).squeeze(-1)
        reset_ids = reset.nonzero(as_tuple=False).squeeze(-1)

        if len(reaching_ids) > 0:
            self.arm_joint_sub_goal[reaching_ids] = torch.lerp(
                self.arm_joint_start[reaching_ids],
                self.arm_joint_goal[reaching_ids],
                (self.timer / self.traj_timesteps)[reaching_ids].reshape(-1, 1),
            )

        if len(holding_ids) > 0:
            self.arm_joint_sub_goal[holding_ids] = self.arm_joint_goal[
                holding_ids
            ].clone()

        if len(reset_ids) > 0:
            self._resample(reset_ids)


class LegJointTrajectoryCommand(CommandTerm):
    """Command term that generates arm joint trajectory for spot arm."""

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
        ----
            cfg: The configuration parameters for the command term.
            env: The environment object.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.leg_joint_idxs = self.robot.find_joints(self.cfg.leg_joint_names)[0]

        # create buffers to store the command
        # -- leg trajectory command
        self.leg_joint_start = torch.zeros(
            self.num_envs, len(self.leg_joint_idxs), device=self.device
        )
        self.leg_joint_goal = torch.zeros(
            self.num_envs, len(self.leg_joint_idxs), device=self.device
        )
        self.leg_joint_sub_goal = torch.zeros(
            self.num_envs, len(self.leg_joint_idxs), device=self.device
        )

        lower_bound = self.robot.data.soft_joint_pos_limits[:, self.leg_joint_idxs, 0]
        upper_bound = self.robot.data.soft_joint_pos_limits[:, self.leg_joint_idxs, 1]
        center = 0.5 * (lower_bound + upper_bound)
        half_range = 0.5 * (upper_bound - lower_bound)
        self.lower_bound = center - self.cfg.ranges.init_range * half_range
        self.upper_bound = center + self.cfg.ranges.init_range * half_range

        self.step_dt = env.step_dt
        self.timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()

        # -- metrics
        self.metrics["leg_joint_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "LegJointTrajectoryCommandCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired arm joints in the environment frame. Shape is (num_envs, 7)."""
        return self.leg_joint_sub_goal

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the arm joint tracking error
        self.metrics["leg_joint_error"] = torch.norm(
            self.robot.data.joint_pos[:, self.leg_joint_idxs] - self.leg_joint_sub_goal,
            dim=1,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        self.leg_joint_start[env_ids] = torch.clamp(
            self.robot.data.joint_pos[env_ids][:, self.leg_joint_idxs],
            self.robot.data.soft_joint_pos_limits[env_ids][:, self.leg_joint_idxs, 0],
            self.robot.data.soft_joint_pos_limits[env_ids][:, self.leg_joint_idxs, 1],
        )
        self.leg_joint_sub_goal[env_ids] = self.leg_joint_start[env_ids]
        self.leg_joint_goal[env_ids] = math_utils.sample_uniform(
            self.lower_bound[env_ids],
            self.upper_bound[env_ids],
            (len(env_ids), len(self.leg_joint_idxs)),
            self.device,
        )

        self.traj_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.timer[env_ids] *= 0.0

    def _update_command(self):
        # step the timer
        self.timer += 1.0

        # update the mid_goal as timer goes
        reaching = self.timer <= self.traj_timesteps
        holding = torch.logical_and(
            self.traj_timesteps < self.timer,
            self.timer <= self.traj_timesteps + self.hold_timesteps,
        )
        reset = self.timer > self.traj_timesteps + self.hold_timesteps

        reaching_ids = reaching.nonzero(as_tuple=False).squeeze(-1)
        holding_ids = holding.nonzero(as_tuple=False).squeeze(-1)
        reset_ids = reset.nonzero(as_tuple=False).squeeze(-1)

        if len(reaching_ids) > 0:
            self.leg_joint_sub_goal[reaching_ids] = torch.lerp(
                self.leg_joint_start[reaching_ids],
                self.leg_joint_goal[reaching_ids],
                (self.timer / self.traj_timesteps)[reaching_ids].reshape(-1, 1),
            )

        if len(holding_ids) > 0:
            self.leg_joint_sub_goal[holding_ids] = self.leg_joint_goal[
                holding_ids
            ].clone()

        if len(reset_ids) > 0:
            self._resample(reset_ids)


class MultiLegJointTrajectoryCommand(CommandTerm):
    """Command term that generates arm joint trajectory for spot arm."""

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
        ----
            cfg: The configuration parameters for the command term.
            env: The environment object.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        # -- leg joint indices
        self.leg_joint_idxs = torch.tensor(  # order: fl, fr, hl, hr
            [
                self.robot.find_joints(names)[0]
                for _, names in self.cfg.leg_joint_names.items()
            ],
            device=self.device,
        )
        self.leg_joint_names = [
            joint_name
            for _, names in self.cfg.leg_joint_names.items()
            for joint_name in self.robot.find_joints(names)[1]
        ]

        # create buffers to store the command
        # -- leg trajectory command
        total_leg_joints = torch.numel(self.leg_joint_idxs)
        self.leg_joint_start = torch.zeros(
            self.num_envs, total_leg_joints, device=self.device
        )
        self.leg_joint_goal = torch.zeros(
            self.num_envs, total_leg_joints, device=self.device
        )
        self.leg_joint_sub_goal = torch.zeros(
            self.num_envs, total_leg_joints, device=self.device
        )
        # -- indices for which leg is PD controlled
        self.command_leg_idxs = torch.zeros(self.num_envs, device=self.device).int()
        # -- indicator for whether any leg is PD controlled
        self.command_leg = torch.ones(self.num_envs, device=self.device).bool()
        # -- indices for the leg joint in command, in the order of fl, fr, hl, hr
        self.command_leg_joint_idxs = torch.arange(
            total_leg_joints, device=self.device
        ).view(len(self.leg_joint_idxs), -1)

        # -- joint limits should be the same for all legs
        assert torch.equal(
            self.robot.data.soft_joint_pos_limits[:, self.leg_joint_idxs[0], :],
            self.robot.data.soft_joint_pos_limits[:, self.leg_joint_idxs[1], :],
        )
        assert torch.equal(
            self.robot.data.soft_joint_pos_limits[:, self.leg_joint_idxs[1], :],
            self.robot.data.soft_joint_pos_limits[:, self.leg_joint_idxs[2], :],
        )
        assert torch.equal(
            self.robot.data.soft_joint_pos_limits[:, self.leg_joint_idxs[2], :],
            self.robot.data.soft_joint_pos_limits[:, self.leg_joint_idxs[3], :],
        )

        lower_bound = self.robot.data.soft_joint_pos_limits[
            :, self.leg_joint_idxs[0], 0
        ]
        upper_bound = self.robot.data.soft_joint_pos_limits[
            :, self.leg_joint_idxs[0], 1
        ]
        center = 0.5 * (lower_bound + upper_bound)
        half_range = 0.5 * (upper_bound - lower_bound)
        self.lower_bound = center - self.cfg.ranges.init_range * half_range
        self.upper_bound = center + self.cfg.ranges.init_range * half_range

        self.step_dt = env.step_dt
        self.timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()

        # -- metrics
        self.metrics["multi_leg_joint_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        msg = "MultiLegJointTrajectoryCommandCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired arm joints in the environment frame. Shape is (num_envs, 12)."""
        return self.leg_joint_sub_goal

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the arm joint tracking error
        batch_indices = torch.arange(self.num_envs).view(-1, 1).repeat(1, 3)
        command_joint_idxs = self.command_leg_joint_idxs[self.command_leg_idxs]
        leg_joint_idxs = self.leg_joint_idxs[self.command_leg_idxs]
        self.metrics["multi_leg_joint_error"] = torch.norm(
            self.robot.data.joint_pos[batch_indices, leg_joint_idxs]
            - self.leg_joint_sub_goal[batch_indices, command_joint_idxs],
            dim=1,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample which leg is PD controlled
        self.command_leg_idxs[env_ids] = torch.randint(
            0, len(self.leg_joint_idxs), (len(env_ids),), device=self.device
        ).int()

        self.leg_joint_start[env_ids] = 0.0
        self.leg_joint_sub_goal[env_ids] = 0.0
        self.leg_joint_goal[env_ids] = 0.0

        batch_indices = env_ids.view(-1, 1).repeat(1, 3)
        command_joint_idxs = self.command_leg_joint_idxs[
            self.command_leg_idxs[env_ids]
        ]  # commanded leg joint idxs in the command
        leg_joint_idxs = self.leg_joint_idxs[
            self.command_leg_idxs[env_ids]
        ]  # commanded leg joint idxs in the simulation

        self.leg_joint_start[batch_indices, command_joint_idxs] = (
            self.robot.data.joint_pos[batch_indices, leg_joint_idxs]
        )
        self.leg_joint_sub_goal[env_ids] = self.leg_joint_start[env_ids]
        self.leg_joint_goal[batch_indices, command_joint_idxs] = (
            math_utils.sample_uniform(
                self.lower_bound[env_ids],
                self.upper_bound[env_ids],
                (len(env_ids), 3),
                self.device,
            )
        )

        # sample some envs to not command the leg, and set the goal to 0 for no command envs
        self.command_leg[env_ids] = (
            torch.rand(len(env_ids), device=self.device) > self.cfg.no_command_leg_prob
        )
        self.leg_joint_start[~self.command_leg] = 0.0
        self.leg_joint_sub_goal[~self.command_leg] = 0.0
        self.leg_joint_goal[~self.command_leg] = 0.0

        self.traj_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.timer[env_ids] *= 0.0

    def _update_command(self):
        # step the timer
        self.timer += 1.0

        # update the mid_goal as timer goes
        reaching = self.timer <= self.traj_timesteps
        holding = torch.logical_and(
            self.traj_timesteps < self.timer,
            self.timer <= self.traj_timesteps + self.hold_timesteps,
        )
        reset = self.timer > self.traj_timesteps + self.hold_timesteps

        reaching_ids = reaching.nonzero(as_tuple=False).squeeze(-1)
        holding_ids = holding.nonzero(as_tuple=False).squeeze(-1)
        reset_ids = reset.nonzero(as_tuple=False).squeeze(-1)

        if len(reaching_ids) > 0:
            self.leg_joint_sub_goal[reaching_ids] = torch.lerp(
                self.leg_joint_start[reaching_ids],
                self.leg_joint_goal[reaching_ids],
                (self.timer / self.traj_timesteps)[reaching_ids].reshape(-1, 1),
            )

        if len(holding_ids) > 0:
            self.leg_joint_sub_goal[holding_ids] = self.leg_joint_goal[
                holding_ids
            ].clone()

        if len(reset_ids) > 0:
            self._resample(reset_ids)


class BasePoseCommand(CommandTerm):
    """Command term that generates arm joint trajectory for spot arm."""

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
        ----
            cfg: The configuration parameters for the command term.
            env: The environment object.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers to store the command
        self.torso_roll_pitch_height_goal = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.torso_projected_gravity_goal = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.gravity_vec = self.robot.data.GRAVITY_VEC_W.clone()

        # -- metrics
        self.metrics["torso_roll_pitch_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["torso_height_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        msg = "BasePoseCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired arm joints in the environment frame. Shape is (num_envs, 12)."""
        return self.torso_roll_pitch_height_goal

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the torso pose tracking error
        self.metrics["torso_roll_pitch_error"] = torch.norm(
            self.torso_projected_gravity_goal[:, :2]
            - self.robot.data.projected_gravity_b[:, :2],
            dim=1,
        )

        self.metrics["torso_height_error"] = torch.abs(
            self.torso_roll_pitch_height_goal[:, 2] - self.robot.data.root_pos_w[:, 2]
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample torso pose command
        r = torch.empty(len(env_ids), device=self.device)
        # -- roll
        self.torso_roll_pitch_height_goal[env_ids, 0] = r.uniform_(
            *self.cfg.ranges.roll
        )
        # -- pitch
        self.torso_roll_pitch_height_goal[env_ids, 1] = r.uniform_(
            *self.cfg.ranges.pitch
        )
        # -- height
        self.torso_roll_pitch_height_goal[env_ids, 2] = r.uniform_(
            *self.cfg.ranges.height
        )

        # pre-compute torso_projected_gravity_goal
        # https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/go1_gym/envs/rewards/corl_rewards.py#L148C17-L148C36
        quat_roll = math_utils.quat_from_angle_axis(
            self.torso_roll_pitch_height_goal[env_ids, 0], self.x_axis
        )
        quat_pitch = math_utils.quat_from_angle_axis(
            self.torso_roll_pitch_height_goal[env_ids, 1], self.y_axis
        )
        desired_base_quat = math_utils.quat_mul(quat_roll, quat_pitch)
        self.torso_projected_gravity_goal[env_ids] = math_utils.quat_rotate_inverse(
            desired_base_quat, self.gravity_vec[env_ids]
        )

    def _update_command(self):
        pass


class ArmLegJointBasePoseCommand(CommandTerm):
    """Command term that generates arm joint trajectory for spot arm."""

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
        ----
            cfg: The configuration parameters for the command term.
            env: The environment object.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.arm_joint_idxs = self.robot.find_joints(self.cfg.arm_joint_names)[0]
        self.leg_joint_idxs = torch.tensor(  # order: fl, fr, hl, hr
            [
                self.robot.find_joints(names)[0]
                for _, names in self.cfg.leg_joint_names.items()
            ],
            device=self.device,
        )
        self.leg_joint_names = [
            joint_name
            for _, names in self.cfg.leg_joint_names.items()
            for joint_name in self.robot.find_joints(names)[1]
        ]

        # create buffers to store the command
        # -- arm trajectory command
        self.arm_joint_start = torch.zeros(
            self.num_envs, len(self.arm_joint_idxs), device=self.device
        )
        self.arm_joint_goal = torch.zeros(
            self.num_envs, len(self.arm_joint_idxs), device=self.device
        )
        self.arm_joint_sub_goal = torch.zeros(
            self.num_envs, len(self.arm_joint_idxs), device=self.device
        )

        # -- leg trajectory command
        self.leg_names = ["fl", "fr", "hl", "hr"]
        total_leg_joints = torch.numel(self.leg_joint_idxs)
        self.leg_joint_start = torch.zeros(
            self.num_envs, total_leg_joints, device=self.device
        )
        self.leg_joint_goal = torch.zeros(
            self.num_envs, total_leg_joints, device=self.device
        )
        self.leg_joint_sub_goal = torch.zeros(
            self.num_envs, total_leg_joints, device=self.device
        )
        # -- indices for which leg is PD controlled
        self.command_leg_idxs = torch.zeros(self.num_envs, device=self.device).int()
        # -- indicator for whether any leg is PD controlled
        self.command_leg = torch.ones(self.num_envs, device=self.device).bool()
        # -- indices for the leg joint in command, in the order of fl, fr, hl, hr
        self.command_leg_joint_idxs = torch.arange(
            total_leg_joints, device=self.device
        ).view(len(self.leg_joint_idxs), -1)

        # -- torso pose command
        self.torso_roll_pitch_height_goal = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.torso_projected_gravity_goal = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.gravity_vec = self.robot.data.GRAVITY_VEC_W.clone()

        # timer
        self.step_dt = env.step_dt
        self.timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()

        # Pre-compute feasible commands
        self.num_cached_command = 500000
        self.cached_command: dict[str, torch.Tensor] = dict()
        self.cached_command["roll"] = math_utils.sample_uniform(
            self.cfg.command_range_roll[0],
            self.cfg.command_range_roll[1],
            (self.num_cached_command,),
            device=self.device,
        )
        self.cached_command["pitch"] = math_utils.sample_uniform(
            self.cfg.command_range_pitch[0],
            self.cfg.command_range_pitch[1],
            (self.num_cached_command,),
            device=self.device,
        )
        self.cached_command["height"] = math_utils.sample_uniform(
            self.cfg.command_range_height[0],
            self.cfg.command_range_height[1],
            (self.num_cached_command,),
            device=self.device,
        )
        self.cfg.command_range_arm_joint = torch.tensor(
            self.cfg.command_range_arm_joint, device=self.device
        )
        self.cached_command["arm_joint"] = math_utils.sample_uniform(
            self.cfg.command_range_arm_joint[0],
            self.cfg.command_range_arm_joint[1],
            (self.num_cached_command, len(self.cfg.command_range_arm_joint[0])),
            device=self.device,
        )
        self.cfg.command_range_leg_joint = torch.tensor(
            self.cfg.command_range_leg_joint, device=self.device
        )
        self.cached_command["sampled_leg_joint"] = math_utils.sample_uniform(
            self.cfg.command_range_leg_joint[0],
            self.cfg.command_range_leg_joint[1],
            (self.num_cached_command, len(self.cfg.command_range_leg_joint[0])),
            device=self.device,
        )
        self.cached_command["sampled_leg_name"] = torch.randint(
            -1, 4, (self.num_cached_command,), device=self.device
        )

        # filter the commands based on command_which_leg
        assert self.cfg.command_which_leg in [-1, 0, 1, 2, 3, 4]
        if self.cfg.command_which_leg != 4:  # mix all leg command
            valid_command_mask = (
                self.cached_command["sampled_leg_name"] == self.cfg.command_which_leg
            )
            self.num_cached_command = valid_command_mask.sum().item()
            for k, v in self.cached_command.items():
                self.cached_command[k] = v[valid_command_mask]

        # -- metrics
        self.metrics["arm_joint_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["leg_joint_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_roll_pitch_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["base_height_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

        self.feet_body_ids = torch.tensor([12, 16, 20, 24], device=self.device).repeat(
            self.env.num_envs, 1
        )
        self.feet_index = torch.arange(4, device=env.device).repeat(
            self.env.num_envs, 1
        )
        self.batch_indices = torch.arange(self.num_envs).view(-1, 1).repeat(1, 3)

    def __str__(self) -> str:
        msg = "ArmLegJointBasePoseCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired arm joints in the environment frame. Shape is (num_envs, 22)."""
        return torch.cat(
            [
                self.arm_joint_sub_goal,
                self.leg_joint_sub_goal,
                self.torso_roll_pitch_height_goal,
            ],
            dim=1,
        )

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # -- compute the arm joint tracking error
        self.metrics["arm_joint_error"] = torch.norm(
            self.robot.data.joint_pos[:, self.arm_joint_idxs] - self.arm_joint_sub_goal,
            dim=1,
        )
        # -- compute the arm joint tracking error
        batch_indices = torch.arange(self.num_envs).view(-1, 1).repeat(1, 3)
        command_joint_idxs = self.command_leg_joint_idxs[self.command_leg_idxs]
        leg_joint_idxs = self.leg_joint_idxs[self.command_leg_idxs]
        self.metrics["leg_joint_error"] = torch.norm(
            self.robot.data.joint_pos[batch_indices, leg_joint_idxs]
            - self.leg_joint_sub_goal[batch_indices, command_joint_idxs],
            dim=1,
        )
        # -- compute the base pose tracking error
        self.metrics["base_roll_pitch_error"] = torch.norm(
            self.torso_projected_gravity_goal[:, :2]
            - self.robot.data.projected_gravity_b[:, :2],
            dim=1,
        )
        self.metrics["base_height_error"] = torch.abs(
            self.torso_roll_pitch_height_goal[:, 2] - self.robot.data.root_pos_w[:, 2]
        )

    def _resample_arm_command(
        self, env_ids: Sequence[int], sampled_command_idx: Sequence[int]
    ):
        self.arm_joint_start[env_ids] = torch.clamp(
            self.robot.data.joint_pos[env_ids][:, self.arm_joint_idxs],
            self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_idxs, 0],
            self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_idxs, 1],
        )
        self.arm_joint_sub_goal[env_ids] = self.arm_joint_start[env_ids]
        self.arm_joint_goal[env_ids] = self.cached_command["arm_joint"][
            sampled_command_idx
        ]

    def _resample_leg_command(
        self, env_ids: Sequence[int], sampled_command_idx: Sequence[int]
    ):
        self.command_leg_idxs[env_ids] = self.cached_command["sampled_leg_name"][
            sampled_command_idx
        ].int()
        self.command_leg_idxs[self.command_leg_idxs == -1] = (
            0  # get a dummy placeholder for no-PD-leg envs
        )

        self.leg_joint_start[env_ids] = 0.0
        self.leg_joint_sub_goal[env_ids] = 0.0
        self.leg_joint_goal[env_ids] = 0.0

        batch_indices = env_ids.view(-1, 1).repeat(1, 3)
        command_joint_idxs = self.command_leg_joint_idxs[
            self.command_leg_idxs[env_ids]
        ]  # commanded leg joint idxs in the command
        leg_joint_idxs = self.leg_joint_idxs[
            self.command_leg_idxs[env_ids]
        ]  # commanded leg joint idxs in the simulation

        self.leg_joint_start[batch_indices, command_joint_idxs] = (
            self.robot.data.joint_pos[batch_indices, leg_joint_idxs].clone()
        )
        self.leg_joint_sub_goal[env_ids] = self.leg_joint_start[env_ids].clone()
        self.leg_joint_goal[batch_indices, command_joint_idxs] = self.cached_command[
            "sampled_leg_joint"
        ][sampled_command_idx]

        self.command_leg[env_ids] = (
            self.cached_command["sampled_leg_name"][sampled_command_idx].int() != -1
        )
        self.leg_joint_start[~self.command_leg] = 0.0
        self.leg_joint_sub_goal[~self.command_leg] = 0.0
        self.leg_joint_goal[~self.command_leg] = 0.0

    def _resample_base_command(
        self, env_ids: Sequence[int], sampled_command_idx: Sequence[int]
    ):
        self.torso_roll_pitch_height_goal[env_ids, 0] = self.cached_command["roll"][
            sampled_command_idx
        ]
        self.torso_roll_pitch_height_goal[env_ids, 1] = self.cached_command["pitch"][
            sampled_command_idx
        ]
        self.torso_roll_pitch_height_goal[env_ids, 2] = self.cached_command["height"][
            sampled_command_idx
        ]

        # pre-compute torso_projected_gravity_goal
        # https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/go1_gym/envs/rewards/corl_rewards.py#L148C17-L148C36
        quat_roll = math_utils.quat_from_angle_axis(
            self.torso_roll_pitch_height_goal[env_ids, 0], self.x_axis
        )
        quat_pitch = math_utils.quat_from_angle_axis(
            self.torso_roll_pitch_height_goal[env_ids, 1], self.y_axis
        )
        desired_base_quat = math_utils.quat_mul(quat_roll, quat_pitch)
        self.torso_projected_gravity_goal[env_ids] = math_utils.quat_rotate_inverse(
            desired_base_quat, self.gravity_vec[env_ids]
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample goal from the cached commands
        sampled_command_idx = torch.randint(
            0, self.num_cached_command, (len(env_ids),), device=self.device
        )

        # -- resample buffers for arm
        self._resample_arm_command(env_ids, sampled_command_idx)
        # -- resample buffers for leg
        self._resample_leg_command(env_ids, sampled_command_idx)
        # -- reset base pose
        self._resample_base_command(env_ids, sampled_command_idx)

        # reset timer
        self.traj_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.timer[env_ids] *= 0.0

    def _update_command(self):
        # step the timer
        self.timer += 1.0

        # update the mid_goal as timer goes
        reaching = self.timer <= self.traj_timesteps
        holding = torch.logical_and(
            self.traj_timesteps < self.timer,
            self.timer <= self.traj_timesteps + self.hold_timesteps,
        )
        reset = self.timer > self.traj_timesteps + self.hold_timesteps

        reaching_ids = reaching.nonzero(as_tuple=False).squeeze(-1)
        holding_ids = holding.nonzero(as_tuple=False).squeeze(-1)
        reset_ids = reset.nonzero(as_tuple=False).squeeze(-1)

        if len(reaching_ids) > 0:
            self.arm_joint_sub_goal[reaching_ids] = torch.lerp(
                self.arm_joint_start[reaching_ids],
                self.arm_joint_goal[reaching_ids],
                (self.timer / self.traj_timesteps)[reaching_ids].reshape(-1, 1),
            )
            self.leg_joint_sub_goal[reaching_ids] = torch.lerp(
                self.leg_joint_start[reaching_ids],
                self.leg_joint_goal[reaching_ids],
                (self.timer / self.traj_timesteps)[reaching_ids].reshape(-1, 1),
            )

        if len(holding_ids) > 0:
            self.arm_joint_sub_goal[holding_ids] = self.arm_joint_goal[
                holding_ids
            ].clone()
            self.leg_joint_sub_goal[holding_ids] = self.leg_joint_goal[
                holding_ids
            ].clone()

        if len(reset_ids) > 0:
            self._resample(reset_ids)
