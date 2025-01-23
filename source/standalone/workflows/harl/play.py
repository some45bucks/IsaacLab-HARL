"""Train an algorithm."""
import argparse
import sys
import json
import time
import numpy as np
import tensorboardX
import torch
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")
parser.add_argument(
        "--algorihm",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_env_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--dir", type=str, default=None, help="folder with trained models")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import random
from datetime import datetime
from harl.runners import RUNNER_REGISTRY

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

import tensorboardX

agent_cfg_entry_point = "harl_ppo_cfg_entry_point"

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    args = args_cli.__dict__

    args['env'] = 'isaaclab'
    args['algo'] = args['algorihm']
    args["exp_name"] = 'play'

    algo_args = agent_cfg

    algo_args['eval']['use_eval'] = False
    algo_args['render']['use_render'] = True
    algo_args['train']['model_dir'] = args['dir']

    env_args = {}
    env_cfg.scene.num_envs = args['num_envs']
    env_args['task'] = args['task']
    env_args['config'] = env_cfg
    env_args['video_settings'] = {}
    env_args['video_settings']['video'] = False

    #create runner
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    
    obs, _, _ = runner.env.reset()
    actions = np.zeros((args['num_envs'],runner.num_agents, runner.env.action_space[0].shape[0]))
    rnn_states = np.zeros(  
        (
            args['num_envs'],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    masks = np.ones(
            (args['num_envs'], runner.num_agents, 1),
            dtype=np.float32,
        )

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            for agent_id in range(runner.num_agents):
                action, _, rnn_state = runner.actor[agent_id].get_actions(obs[:,agent_id,:],rnn_states[:,agent_id,:],masks[:,agent_id,:],None,None)

                actions[:, agent_id, :] = action.cpu().numpy()
                rnn_states[:, agent_id, :] = rnn_state.cpu().numpy()

            obs, _, _, dones, _, _ = runner.env.step(actions)
            dones_env = np.all(dones, axis=1)
            masks = np.ones((args['num_envs'], runner.num_agents, 1),dtype=np.float32,)
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), runner.num_agents, 1), dtype=np.float32)
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(),runner.num_agents,runner.recurrent_n,runner.rnn_hidden_size),dtype=np.float32)

    runner.env.close()




if __name__ == "__main__":
    main()
    simulation_app.close()
