"""Train an algorithm."""
import argparse
import sys
import json
import time
import numpy as np
import tensorboardX
import torch
from omni.isaac.lab.app import AppLauncher
from pynput.keyboard import Key, Listener
import threading


parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")
parser.add_argument(
        "--algorithm",
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
    move_command = 'stay_still'

    movements=dict(
        rotate_left=torch.tensor([0,0,1]),
        rotate_right=torch.tensor([0,0,-1]),
        move_left=torch.tensor([0,1,0]),
        move_right=torch.tensor([0,-1,0]),
        move_forward=torch.tensor([1,0,0]),
        move_backward=torch.tensor([-1,0,0]),
        stay_still=torch.tensor([0,0,0])
    )
    move_vector = movements[move_command]

    def parse_input(key):
        nonlocal move_command
        nonlocal movements
        nonlocal move_vector

        if key == Key.up:
            move_vector += movements['move_forward']
        elif key == Key.left:
            move_vector += movements['move_left']
        elif key == Key.right:
            move_vector += movements['move_right']
        elif key == Key.down:
            move_vector += movements['move_backward']
        elif hasattr(key, 'char'):
            if key.char == 'a':
                move_vector += movements['rotate_left']
            elif key.char == 'd':
                move_vector += movements['rotate_right']

        move_vector = torch.clip(move_vector, -1, 1)

    def set_to_no_move(key):
        nonlocal move_vector
        move_vector = torch.tensor([0,0,0])
    
    listener = Listener(on_press=parse_input, on_release=set_to_no_move)
    listener_thread = threading.Thread(target=listener.start, daemon=True)
    listener_thread.start()

    args = args_cli.__dict__

    args['env'] = 'isaaclab'
    args['algo'] = args['algorithm']
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

    max_action_space = 0

    for agent_id, obs_space in runner.env.action_space.items():
        if obs_space.shape[0] > max_action_space:
            max_action_space = obs_space.shape[0]
        

    actions = np.zeros((args['num_envs'],runner.num_agents, max_action_space), dtype=np.float64)
    rnn_states = np.zeros(  
        (
            args['num_envs'],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float64,
    )
    masks = np.ones(
            (args['num_envs'], runner.num_agents, 1),
            dtype=np.float64,
        )

    # simulate environment
    total_rewards = np.zeros((args['num_envs'], runner.num_agents,1), dtype=np.float64)
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            for agent_id in range(runner.num_agents):
                action, _, rnn_state = runner.actor[agent_id].get_actions(obs[:,agent_id,:],rnn_states[:,agent_id,:],masks[:,agent_id,:],None,None)
                action_space = action.shape[1]
                actions[:, agent_id, :action_space] = action.cpu().numpy()
                rnn_states[:, agent_id, :] = rnn_state.cpu().numpy()

            runner.env.unwrapped._commands[:,:] = move_vector

            obs, _, rewards, dones, _, _ = runner.env.step(actions)

            total_rewards += rewards
            print(f"Average reward: {rewards.mean(axis=0)}")
            dones_env = np.all(dones, axis=1)
            masks = np.ones((args['num_envs'], runner.num_agents, 1),dtype=np.float64,)
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), runner.num_agents, 1), dtype=np.float64)
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(),runner.num_agents,runner.recurrent_n,runner.rnn_hidden_size),dtype=np.float64)

    runner.env.close()




if __name__ == "__main__":
    main()
    simulation_app.close()
