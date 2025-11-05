# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    weight: float,
    num_steps: int,
):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
    ----
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.

    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(
        env.reward_manager.get_term_cfg(term_name).weight, device=env.device
    )


def modify_contact_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    body_names: str,
    num_steps: int,
):
    """Curriculum that modifies the termination with a given number of steps.

    Args:
    ----
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.

    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params["sensor_cfg"].body_names = body_names
        term_cfg.params["sensor_cfg"].body_ids = slice(None, None, None)
        term_cfg.params["sensor_cfg"].resolve(env.scene)
        env.termination_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(
        len(
            env.termination_manager.get_term_cfg(term_name)
            .params["sensor_cfg"]
            .body_ids
        ),
        device=env.device,
    )
