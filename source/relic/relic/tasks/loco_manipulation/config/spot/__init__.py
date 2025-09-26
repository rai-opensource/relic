# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import gymnasium as gym

from . import agents, spot_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Spot-Interlimb-Phase-1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotInterlimbEnvCfg_Phase_1,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SpotInterlimbPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Spot-Interlimb-Phase-2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotInterlimbEnvCfg_Phase_2,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SpotInterlimbPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Spot-Interlimb-Phase-3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotInterlimbEnvCfg_Phase_3,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SpotInterlimbPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Spot-Interlimb-Phase-4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotInterlimbEnvCfg_Phase_4,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SpotInterlimbPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Spot-Interlimb-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_env_cfg.SpotInterlimbEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.SpotInterlimbPPORunnerCfg,
    },
)
