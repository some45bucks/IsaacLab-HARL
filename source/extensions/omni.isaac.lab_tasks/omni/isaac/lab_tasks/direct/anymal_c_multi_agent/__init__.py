# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .anymal_c_multi_agent import AnymalCMultiAgent, AnymalCMultiAgentFlatEnvCfg, AnymalCMultiAgentRoughEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.anymal_c_multi_agent:AnymalCMultiAgent",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCMultiAgentFlatEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_ppo_cfg_entry_point": f"{agents.__name__}:harl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Multi-Agent-Velocity-Rough-Anymal-C-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.anymal_c_multi_agent:AnymalCMultiAgent",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCMultiAgentRoughEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
