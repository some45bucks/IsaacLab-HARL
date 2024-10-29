# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an Isaac Lab environment instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # or ml_framework="jax"

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env  # for PyTorch, or...
    from skrl.envs.jax.wrappers import wrap_env    # for JAX

    env = wrap_env(env, wrapper="isaaclab")

"""

# needed to import for type hinting: Agent | list[Agent]
from __future__ import annotations

from typing import Literal

from omni.isaac.lab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv

"""
Vectorized environment wrapper.
"""


def HarlEnvWrapper(
    env: ManagerBasedRLEnv | DirectRLEnv | DirectMARLEnv,
    wrapper: Literal["auto", "isaaclab", "isaaclab-single-agent", "isaaclab-multi-agent"] = "isaaclab",
):
    
    if (
        not isinstance(env.unwrapped, ManagerBasedRLEnv)
        and not isinstance(env.unwrapped, DirectRLEnv)
        and not isinstance(env.unwrapped, DirectMARLEnv)
    ):
        raise ValueError(
            "The environment must be inherited from ManagerBasedRLEnv, DirectRLEnv or DirectMARLEnv. Environment type:"
            f" {type(env)}"
        )

    from skrl.envs.wrappers.torch import wrap_env

    # wrap and return the environment
    return wrap_env(env, wrapper)
