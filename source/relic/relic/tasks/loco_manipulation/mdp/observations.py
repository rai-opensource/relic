# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Functions specific to the loco-manipulation environments."""

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv


def known_external_force_torque(
    env: ManagerBasedRLEnv,
    event_name: str,
    scale: bool,
) -> torch.Tensor:
    """Known external force torque amount."""
    # event_manager is established after first call
    if not hasattr(env, "event_manager"):
        return torch.zeros((env.num_envs, 3), device=env.device)

    else:
        event_term = env.event_manager.get_term_cfg(event_name).func
        external_forces = event_term.external_forces.view(env.num_envs, -1)
        if scale:
            external_forces = math_utils.scale_transform(
                external_forces, event_term.force_range[0], event_term.force_range[1]
            )
        return external_forces


def gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    if not hasattr(env, "reward_manager"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    else:
        steps = env.reward_manager.get_term_cfg("gait").func.steps
        command_leg = env.command_manager.get_term(
            "arm_leg_joint_base_pose"
        ).command_leg
        max_length = (
            command_leg
            * env.reward_manager.get_term_cfg("gait").func.three_leg_phase_len
            + (1 - command_leg.int())
            * env.reward_manager.get_term_cfg("gait").func.four_leg_phase_len
        )
        phase = steps % max_length
        phase = math_utils.scale_transform(
            phase, torch.zeros_like(max_length), max_length
        )
        return phase.view(-1, 1)
