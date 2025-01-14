# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""



import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
# import cli_args  # isort: skip
# source/standalone/workflows/rsl_rl/cli_args.py
import source.standalone.workflows.rsl_rl.cli_args as cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import omni.isaac.lab.sim as sim_utils
import numpy as np

def design_scene_with_assets(cfg_assets, asset_count=100, spawn_area=(-50, 50, -100, 100)):
    """
    Adds assets like vegetation or rocks to the scene based on USD paths and probability distribution,
    and places them using the provided `func` method.
    
    Args:
        cfg_assets (dict): Dictionary where keys are the asset names and values contain asset configuration, 
                           including USD paths and the percentage chance of appearance, and a function to place them.
        asset_count (int): Number of assets to add to the scene.
        spawn_area (tuple): Tuple defining the spawn area as (x_min, x_max, y_min, y_max).
    """
    # Unpack the spawn area tuple
    x_min, x_max, y_min, y_max = spawn_area

    # Generate random locations for assets within the specified spawn area
    asset_locs = np.random.uniform([x_min, y_min], [x_max, y_max], (asset_count, 2))
    
    # Generate random probabilities for each asset placement
    asset_prob = np.random.rand(asset_count)

    # Extract asset probabilities and normalize them
    asset_types = list(cfg_assets.keys())
    probabilities = np.array([cfg_assets[asset]['probability'] for asset in asset_types])
    normalized_probabilities = probabilities / probabilities.sum()
    
    # Cumulative probability for selecting asset type
    cumul_prob = np.cumsum(normalized_probabilities)

    # Place the assets in the scene
    for i in range(asset_count):
        # Determine which asset to place based on probability
        for j, asset_type in enumerate(asset_types):
            if asset_prob[i] <= cumul_prob[j]:
                # Place the asset at the given location using the `func` method
                cfg_asset = cfg_assets[asset_type]['config']
                cfg_asset.func(f"/World/{asset_type}_{i}", cfg_asset, translation=(asset_locs[i][0], asset_locs[i][1], 0.0))
                break


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Example configuration for assets
    cfg_assets = {
        'tree': {'config': sim_utils.UsdFileCfg(usd_path="tile_forest/tree.usd"),'probability': 0.25},
        'bush': {'config': sim_utils.UsdFileCfg(usd_path="tile_forest/bush.usd"), 'probability': 0.45},
        'fern': {'config': sim_utils.UsdFileCfg(usd_path="tile_forest/fern.usd"), 'probability': 0.3},
    }
    # env.unwrapped.scene._terrain.meshes
    design_scene_with_assets(cfg_assets, asset_count=1000)


    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
