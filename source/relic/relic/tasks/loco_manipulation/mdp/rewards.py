# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import RewardTermCfg
from isaaclab.envs import ManagerBasedRLEnv


# -- Task Rewards


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    ramp_at_vel: float = 1.0,
    ramp_rate: float = 0.5,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm(
        (target - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(
        1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0
    )
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


def base_angular_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm(
        (target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1
    )
    return torch.exp(-ang_vel_error / std)


def arm_joint_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, command_name: str
) -> torch.Tensor:
    """Reward tracking of arm joint commands using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command(command_name)
    arm_joint_error = torch.linalg.norm(
        (target - asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    return torch.exp(-arm_joint_error / std)


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(
        current_contact_time - current_air_time, -mode_time, mode_time
    )
    cmd = (
        torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        .unsqueeze(dim=1)
        .expand(-1, 4)
    )
    body_vel = (
        torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        .unsqueeze(dim=1)
        .expand(-1, 4)
    )
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def adaptive_air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    four_leg_cycle_time: float,
    three_leg_cycle_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    leg_not_in_command = ~env.command_manager.get_term(
        "arm_leg_joint_base_pose"
    ).command_leg
    vel_cmd = (
        torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        .unsqueeze(dim=1)
        .expand(-1, 4)
    )
    body_vel = (
        torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        .unsqueeze(dim=1)
        .expand(-1, 4)
    )

    four_leg_air_mode_time = four_leg_contact_mode_time = four_leg_cycle_time / 2
    three_leg_air_mode_time, three_leg_contact_mode_time = (
        three_leg_cycle_time / 3,
        2 * three_leg_cycle_time / 3,
    )
    air_mode_time = (
        four_leg_air_mode_time * leg_not_in_command
        + three_leg_air_mode_time * (1 - leg_not_in_command.int())
    ).view(-1, 1)
    contact_mode_time = (
        four_leg_contact_mode_time * leg_not_in_command
        + three_leg_contact_mode_time * (1 - leg_not_in_command.int())
    ).view(-1, 1)

    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]
    foot_in_air = current_air_time > 0.0
    foot_in_contact = current_contact_time > 0.0

    t_air_min = torch.clip(current_air_time, max=air_mode_time)
    t_contact_min = torch.clip(current_contact_time, max=contact_mode_time)

    air_time_reward = torch.where(current_air_time < air_mode_time, t_air_min, 0.0)
    contact_time_reward = torch.where(
        current_contact_time < contact_mode_time, t_contact_min, 0.0
    )
    time_reward = foot_in_air * air_time_reward + foot_in_contact * contact_time_reward

    stance_cmd_reward = torch.clip(
        current_contact_time - current_air_time,
        -four_leg_contact_mode_time,
        four_leg_contact_mode_time,
    )

    reward = torch.where(
        torch.logical_or(vel_cmd > 0.0, body_vel > velocity_threshold),
        time_reward,
        stance_cmd_reward,
    )

    return torch.sum(reward, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
        ----
            cfg: The configuration of the reward.
            env: The RL environment instance.

        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[
            cfg.params["sensor_cfg"].name
        ]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError(
                "This reward only supports gaits with two pairs of synchronized feet, like trotting."
            )
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[
            0
        ]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[
            0
        ]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
        ----
            env: The RL environment instance.

        Returns:
        -------
            The reward value.

        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(
            self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1]
        )
        sync_reward_1 = self._sync_reward_func(
            self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1]
        )
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(
            self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0]
        )
        async_reward_1 = self._async_reward_func(
            self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1]
        )
        async_reward_2 = self._async_reward_func(
            self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1]
        )
        async_reward_3 = self._async_reward_func(
            self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1]
        )
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0 and leg not commanded
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        leg_not_in_command = ~env.command_manager.get_term(
            "arm_leg_joint_base_pose"
        ).command_leg
        return torch.where(
            torch.logical_and(cmd > 0.0, leg_not_in_command),
            sync_reward * async_reward,
            0.0,
        )
        # body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1) # TODO is this needed?
        # return torch.where(
        #     torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        # )

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(
            torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2
        )
        se_contact = torch.clip(
            torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]),
            max=self.max_err**2,
        )
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(
            torch.square(air_time[:, foot_0] - contact_time[:, foot_1]),
            max=self.max_err**2,
        )
        se_act_1 = torch.clip(
            torch.square(contact_time[:, foot_0] - air_time[:, foot_1]),
            max=self.max_err**2,
        )
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


# -- Regularization Rewards


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm(
        (env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def air_time_variance_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def adaptive_air_time_variance_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def base_motion_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm(
        (asset.data.projected_gravity_b[:, :2]), dim=1
    ) + torch.square(
        asset.data.projected_gravity_b[:, 2]
        + torch.ones_like(asset.data.projected_gravity_b[:, 2])
    )


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
        )[0]
        > threshold
    )
    foot_planar_velocity = torch.linalg.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2
    )

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm(
        (
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        reward,
        stand_still_scale * reward,
    )


def all_leg_flight_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Compute the full flight state for each environment.

    Args:
    ----
        env (ManagerBasedRLEnv): _description_
        asset_cfg (SceneEntityCfg): _description_
        sensor_cfg (SceneEntityCfg): _description_
        threshold (float): A body is in contact is the sensed force is above this threshold [N].

    Returns:
    -------
        torch.Tensor: A tensor with shape (n_envs x 1) representing the full flight state.

    """
    # extract the used quantities (to enable type-hinting)
    # TODO avoid computing for command leg
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids

    # check if contact force is above threshold
    foot_contact_forces = contact_sensor.data.net_forces_w[:, body_ids, :].norm(dim=-1)
    is_flight = foot_contact_forces < threshold
    is_full_flight_phase = torch.all(is_flight, dim=-1)

    return is_full_flight_phase.float()


def three_leg_flight_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Compute the flight state for each environment.

    Args:
    ----
        env (ManagerBasedRLEnv): _description_
        asset_cfg (SceneEntityCfg): _description_
        sensor_cfg (SceneEntityCfg): _description_
        threshold (float): A body is in contact is the sensed force is above this threshold [N].

    Returns:
    -------
        torch.Tensor: A tensor with shape (n_envs x 1) representing the full flight state.

    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids

    # check if contact force is above threshold
    foot_contact_forces = contact_sensor.data.net_forces_w[:, body_ids, :].norm(dim=-1)
    foot_is_flight = foot_contact_forces < threshold
    num_foot_is_flight = foot_is_flight.sum(dim=1)
    three_leg_flight = num_foot_is_flight >= 3

    return three_leg_flight.float()


def foot_impact_penalty_2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize foot impact when coming into contact with the ground"""
    asset: Articulation = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    foot_down_velocity = torch.clamp(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2], max=0.0
    )
    reward = first_contact * torch.square(foot_down_velocity)
    return torch.sum(reward, dim=1)


def joint_energy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    energy = torch.sum(
        torch.abs(
            asset.data.applied_torque[:, asset_cfg.joint_ids]
            * asset.data.joint_vel[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return torch.exp(-energy / std**2)


def track_base_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    torso_projected_gravity_goal = env.command_manager.get_term(
        command_name
    ).torso_projected_gravity_goal

    roll_pitch_error = torch.norm(
        torso_projected_gravity_goal[:, :2] - asset.data.projected_gravity_b[:, :2],
        dim=1,
    )
    return torch.exp(-roll_pitch_error / std)


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    torso_height_goal = env.command_manager.get_term(
        command_name
    ).torso_roll_pitch_height_goal[:, 2]

    height_error = torch.abs(torso_height_goal - asset.data.root_pos_w[:, 2])
    return torch.exp(-height_error / std)


def track_base_orientation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2-kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    torso_projected_gravity_goal = env.command_manager.get_term(
        command_name
    ).torso_projected_gravity_goal

    return torch.sum(
        torch.square(
            torso_projected_gravity_goal[:, :2] - asset.data.projected_gravity_b[:, :2]
        ),
        dim=1,
    )


def track_base_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2-kernel.

    Note:
    ----
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    torso_height_goal = env.command_manager.get_term(
        command_name
    ).torso_roll_pitch_height_goal[:, 2]

    return torch.square(asset.data.root_pos_w[:, 2] - torso_height_goal)


def adaptive_joint_torques_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    no_leg_ctl_weight: float,
    leg_ctl_weight: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint
    torques contribute to the L2 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_torques = torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1
    )
    leg_is_commanded = env.command_manager.get_term(command_name).command_leg
    weight = torch.where(leg_is_commanded, leg_ctl_weight, no_leg_ctl_weight)
    return weight * joint_torques


def adaptive_joint_acc_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    no_leg_ctl_weight: float,
    leg_ctl_weight: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint
    accelerations contribute to the L2 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = torch.sum(
        torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1
    )
    leg_is_commanded = env.command_manager.get_term(command_name).command_leg
    weight = torch.where(leg_is_commanded, leg_ctl_weight, no_leg_ctl_weight)
    return weight * joint_acc


def adaptive_action_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    no_leg_ctl_weight: float,
    leg_ctl_weight: float,
) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2-kernel."""
    action_rate = torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )
    leg_is_commanded = env.command_manager.get_term(command_name).command_leg
    weight = torch.where(leg_is_commanded, leg_ctl_weight, no_leg_ctl_weight)
    return weight * action_rate


def feet_air_time_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    three_leg_target_time: float,
    four_leg_target_time: float,
    cycle_time: float,
    air_std: float,
    contact_std: float,
    std: float,
) -> torch.Tensor:
    """Reward steps taken by the feet using exp-kernel.

    This function rewards the agent for taking steps that are close to a target. This helps ensure
    that the robot lifts its feet off the ground and takes steps for certain time. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command_leg = env.command_manager.get_term(
        "arm_leg_joint_base_pose"
    ).command_leg.int()

    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    target_air_time = three_leg_target_time * command_leg + four_leg_target_time * (
        1 - command_leg
    )
    air_time_diff = torch.abs(last_air_time - target_air_time.view(-1, 1))
    _air_std = air_std * command_leg + std * (1 - command_leg)
    reward = torch.sum(
        torch.exp(-air_time_diff / _air_std.view(-1, 1)) * first_contact, dim=1
    )

    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    target_contact_time = (cycle_time - three_leg_target_time) * command_leg + (
        cycle_time - four_leg_target_time
    ) * (1 - command_leg)
    contact_time_diff = torch.abs(last_contact_time - target_contact_time.view(-1, 1))
    _contact_std = contact_std * command_leg + std * (1 - command_leg)
    reward += torch.sum(
        torch.exp(-contact_time_diff / _contact_std.view(-1, 1)) * first_air, dim=1
    )

    # no reward for zero command TODO if zero command, force the robot for stance
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    )
    return reward


def dof_torque_limits_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    over_limit = torch.norm(
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        - asset.data.computed_torque[:, asset_cfg.joint_ids],
        dim=1,
    )
    return over_limit


class ThreeLegGaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
        ----
            cfg: The configuration of the reward.
            env: The RL environment instance.

        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[
            cfg.params["sensor_cfg"].name
        ]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        # match foot body names with corresponding foot body ids
        self.feet_names = cfg.params["feet_names"]
        self.gait_assignment = []
        for vs in cfg.params["gait_assignment"]:
            self.gait_assignment.append(
                [self.contact_sensor.find_bodies(self.feet_names[i])[0][0] for i in vs]
            )
        self.gait_assignment = torch.tensor(self.gait_assignment, device=self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        feet_names,
        gait_assignment,
        gait_cycle_time: float,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
        ----
            env: The RL environment instance.

        Returns:
        -------
            The reward value.

        """
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        vel_cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        leg_in_command = env.command_manager.get_term(
            "arm_leg_joint_base_pose"
        ).command_leg
        leg_in_command_idxs = env.command_manager.get_term(
            "arm_leg_joint_base_pose"
        ).command_leg_idxs
        leg_1_body_id, leg_2_body_id, leg_3_body_id = self.gait_assignment[
            leg_in_command_idxs
        ].T

        # collect air/cotact time
        leg_1_air_time = torch.gather(air_time, 1, leg_1_body_id.view(-1, 1)).view(-1)
        leg_2_air_time = torch.gather(air_time, 1, leg_2_body_id.view(-1, 1)).view(-1)
        leg_3_air_time = torch.gather(air_time, 1, leg_3_body_id.view(-1, 1)).view(-1)
        leg_1_contact_time = torch.gather(
            contact_time, 1, leg_1_body_id.view(-1, 1)
        ).view(-1)
        leg_2_contact_time = torch.gather(
            contact_time, 1, leg_2_body_id.view(-1, 1)
        ).view(-1)
        leg_3_contact_time = torch.gather(
            contact_time, 1, leg_3_body_id.view(-1, 1)
        ).view(-1)

        # compute reward
        leg_1_reward = self._leg_reward_func(
            leg_1_air_time, leg_2_contact_time, leg_3_contact_time, gait_cycle_time
        )  # leg_1_contact_time, leg_2_air_time, leg_3_air_time)
        leg_2_reward = self._leg_reward_func(
            leg_2_air_time, leg_3_contact_time, leg_1_contact_time, gait_cycle_time
        )  # leg_2_contact_time, leg_3_air_time, leg_1_air_time)
        leg_3_reward = self._leg_reward_func(
            leg_3_air_time, leg_1_contact_time, leg_2_contact_time, gait_cycle_time
        )  # leg_3_contact_time, leg_1_air_time, leg_2_air_time)
        leg_reward = leg_1_reward * leg_2_reward * leg_3_reward

        # only enforce gait if cmd > 0 and leg in commanded
        return torch.where(
            torch.logical_and(vel_cmd > 0.0, leg_in_command), leg_reward, 0.0
        )

    def _leg_reward_func_two_way(
        self,
        air_time: torch.Tensor,
        contact_time_1: torch.Tensor,
        contact_time_2: torch.Tensor,
        gait_cycle_time: float,
        contact_time: torch.Tensor,
        air_time_1: torch.Tensor,
        air_time_2: torch.Tensor,
    ) -> torch.Tensor:
        leg_in_air = air_time > 0.0
        # leg 2 contact time = leg 1 air time + gait_cycle_time / 3
        error_1 = torch.clip(
            torch.square(contact_time_1 - (air_time + gait_cycle_time / 3)),
            max=self.max_err**2,
        )
        # leg 3 contact time = leg 1 air time
        error_2 = torch.clip(
            torch.square(contact_time_2 - air_time), max=self.max_err**2
        )
        # leg 1 contact time = leg 2 air time = leg 3 air time = 0.0
        error_3 = torch.clip(torch.square(contact_time), max=self.max_err**2)
        error_4 = torch.clip(torch.square(air_time_1), max=self.max_err**2)
        error_5 = torch.clip(torch.square(air_time_2), max=self.max_err**2)
        error = leg_in_air * (error_1 + error_2 + error_3 + error_4 + error_5)
        return torch.exp(-error / self.std)

    def _leg_reward_func(
        self,
        air_time: torch.Tensor,
        contact_time_1: torch.Tensor,
        contact_time_2: torch.Tensor,
        gait_cycle_time: float,
    ) -> torch.Tensor:
        leg_in_air = air_time > 0.0
        # leg 2 contact time = leg 1 air time + gait_cycle_time / 3
        error_1 = torch.clip(
            torch.square(contact_time_1 - (air_time + gait_cycle_time / 3)),
            max=self.max_err**2,
        )
        # leg 3 contact time = leg 1 air time
        error_2 = torch.clip(
            torch.square(contact_time_2 - air_time), max=self.max_err**2
        )
        # leg 1 contact time = leg 2 air time = leg 3 air time = 0.0
        error = leg_in_air * (error_1 + error_2)
        return torch.exp(-error / self.std)


def track_lin_vel_xy_scale_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    velocity_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target_linvel_b = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.sum(
        torch.square(target_linvel_b - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # scale the reward if velocity is lower than the threshold
    scale = torch.where(
        torch.norm(target_linvel_b, dim=1) < velocity_threshold, 2.0, 1.0
    )
    reward *= scale
    return reward


def track_ang_vel_z_scale_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    velocity_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target_angvel_b = env.command_manager.get_command(command_name)[:, 2]
    ang_vel_error = torch.square(target_angvel_b - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # scale the reward if velocity is lower than the threshold
    scale = torch.where(torch.abs(target_angvel_b) < velocity_threshold, 2.0, 1.0)
    reward *= scale
    return reward
