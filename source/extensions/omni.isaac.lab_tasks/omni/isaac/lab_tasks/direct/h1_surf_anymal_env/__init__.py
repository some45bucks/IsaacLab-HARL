# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .h1_surf_anymal_env import HeterogeneousMultiAgentFlatSurfEnvCfg, HeterogeneousMultiAgentRoughSurfEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Anymal-H1-Surf-Flat-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.h1_anymal_env:HeterogeneousMultiAgentSurf",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousMultiAgentFlatSurfEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_ppo_cfg_entry_point": f"{agents.__name__}:harl_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Anymal-H1-Surf-Rough-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.multi_agent_heterogeneous:HeterogeneousMultiAgentSurf",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousMultiAgentRoughSurfEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
