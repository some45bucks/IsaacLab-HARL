# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_robots", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
print(args_cli)
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from harl.algorithms.actors.happo import HAPPO

def restore(path, env):
    """Restore model parameters."""

    ALGO_ARGS = {
    "model": {
        "hidden_sizes": [128, 128],
        "activation_func": "relu",
        "use_feature_normalization": True,
        "initialization_method": "orthogonal",
        "gain": 0.01,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": False,
        "recurrent_n": 1,
        "data_chunk_length": 10,
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "opti_eps": 0.00001,
        "weight_decay": 0,
        "std_x_coef": 1,
        "std_y_coef": 0.5,
    },
    "algo": {
        "ppo_epoch": 5,
        "critic_epoch": 5,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "actor_num_mini_batch": 1,
        "critic_num_mini_batch": 1,
        "entropy_coef": 0.01,
        "value_loss_coef": 1,
        "use_max_grad_norm": True,
        "max_grad_norm": 10.0,
        "use_gae": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "use_huber_loss": True,
        "use_policy_active_masks": True,
        "huber_delta": 10.0,
        "action_aggregation": "prod",
        "share_param": False,
        "fixed_order": False,
    }
    }
    policy_actor_state_dict = torch.load(
        path
    )

    model = HAPPO({ **ALGO_ARGS['model'],**ALGO_ARGS['algo']}, env.observation_space.shape[1], env.action_space.shape[1], env.device)
    model.actor.load_state_dict(policy_actor_state_dict)
    return model

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        'Isaac-Velocity-Flat-Anymal-C-Direct-v0', 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric,
        num_robots=args_cli.num_robots
    )
    # create environment
    env = gym.make('Isaac-Velocity-Flat-Anymal-C-Direct-v0', cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # load level policy
    policy_path = "/home/isaacp/sharedrepos/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/h1_anymal_env/model/actor_agent0.pt"
    model = restore(policy_path, env)
    # check if policy file exists
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
    file_bytes = read_file(policy_path)
    # jit load the policy
    policy = torch.jit.load(file_bytes).to(env.unwrapped.device).eval()
    # reset environment
    obs, _ = env.reset()
    obs[:,9:12] = [1,0,0]
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = model.get_actions(obs)
            # actions = 2 * torch.rand((args_cli.num_envs*args_cli.num_robots, env.action_space.shape[1]), device=env.unwrapped.device) - 1
            # apply actions
            obs, _, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
