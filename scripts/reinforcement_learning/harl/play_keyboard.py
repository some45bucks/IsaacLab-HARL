# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an algorithm."""

import argparse

# import numpy as np
import sys
import threading
import time
import torch

from pynput.keyboard import Key, Listener

from isaaclab.app import AppLauncher

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

from harl.runners import RUNNER_REGISTRY

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"harl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    keys_pressed = {}
    move_command = "stay_still"
    dt = 0.1
    movements = dict(
        rotate_left=torch.tensor([0.0, 0.0, 1.0]),
        rotate_right=torch.tensor([0.0, 0.0, -1.0]),
        move_left=torch.tensor([0.0, 1.0, 0.0]),
        move_right=torch.tensor([0.0, -1.0, 0.0]),
        move_forward=torch.tensor([1.0, 0.0, 0.0]),
        move_backward=torch.tensor([-1.0, 0.0, 0.0]),
        stay_still=torch.tensor([0.0, 0.0, 0.0]),
    )
    move_vector = movements[move_command]

    def parse_input(key):
        nonlocal move_command
        nonlocal movements
        nonlocal move_vector
        nonlocal keys_pressed

        if hasattr(key, "char"):
            key = key.char

        keys_pressed[key] = True

        # if key == Key.up:
        #     move_vector += movements["move_forward"]
        # elif key == Key.left:
        #     move_vector += movements["move_left"]
        # elif key == Key.right:
        #     move_vector += movements["move_right"]
        # elif key == Key.down:
        #     move_vector += movements["move_backward"]
        # elif hasattr(key, "char"):
        #     if key.char == "a":
        #         move_vector += movements["rotate_left"]
        #     elif key.char == "d":
        #         move_vector += movements["rotate_right"]

        # move_vector = torch.clip(move_vector, -1, 1)

    def set_to_no_move(key):
        nonlocal move_command
        nonlocal movements
        nonlocal move_vector
        nonlocal keys_pressed

        if hasattr(key, "char"):
            key = key.char

        keys_pressed[key] = False

        # if key == Key.up:
        #     move_vector -= movements["move_forward"]
        # elif key == Key.left:
        #     move_vector -= movements["move_left"]
        # elif key == Key.right:
        #     move_vector -= movements["move_right"]
        # elif key == Key.down:
        #     move_vector -= movements["move_backward"]
        # elif hasattr(key, "char"):
        #     if key.char == "a":
        #         move_vector -= movements["rotate_left"]
        #     elif key.char == "d":
        #         move_vector -= movements["rotate_right"]

        # move_vector = torch.clip(move_vector, -1, 1)

    def check_keys_held():
        nonlocal move_command
        nonlocal movements
        nonlocal move_vector
        nonlocal keys_pressed
        nonlocal dt

        while True:
            before_vector = move_vector.clone()

            for key, down in keys_pressed.items():
                if key == Key.up and down:
                    move_vector += movements["move_forward"] * dt
                if key == Key.left and down:
                    move_vector += movements["move_left"] * dt
                if key == Key.right and down:
                    move_vector += movements["move_right"] * dt
                if key == Key.down and down:
                    move_vector += movements["move_backward"] * dt
                if key == "a" and down:
                    move_vector += movements["rotate_left"] * dt
                if key == "d" and down:
                    move_vector += movements["rotate_right"] * dt

            move_vector[move_vector == before_vector] -= move_vector[move_vector == before_vector] * dt

            move_vector = torch.clip(move_vector, -0.5, 0.5)
            time.sleep(dt)

    check_thread = threading.Thread(target=check_keys_held, daemon=True)
    check_thread.start()

    listener = Listener(on_press=parse_input, on_release=set_to_no_move)
    listener_thread = threading.Thread(target=listener.start, daemon=True)
    listener_thread.start()

    args = args_cli.__dict__

    args["env"] = "isaaclab"
    args["algo"] = args["algorithm"]
    args["exp_name"] = "play"

    algo_args = agent_cfg

    algo_args["eval"]["use_eval"] = False
    algo_args["render"]["use_render"] = True
    algo_args["train"]["model_dir"] = args["dir"]

    env_args = {}
    env_cfg.scene.num_envs = args["num_envs"]
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    env_args["video_settings"] = {}
    env_args["video_settings"]["video"] = False

    # create runner
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)

    obs, _, _ = runner.env.reset()

    max_action_space = 0

    for agent_id, obs_space in runner.env.action_space.items():
        if obs_space.shape[0] > max_action_space:
            max_action_space = obs_space.shape[0]

    actions = torch.zeros((args["num_envs"], runner.num_agents, max_action_space), dtype=torch.float64, device="cuda:0")
    rnn_states = torch.zeros(
        (
            args["num_envs"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=torch.float64,
        device="cuda:0",
    )
    masks = torch.ones((args["num_envs"], runner.num_agents, 1), dtype=torch.float64, device="cuda:0")

    # simulate environment
    total_rewards = torch.zeros((args["num_envs"], runner.num_agents, 1), dtype=torch.float64, device="cuda:0")
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            for agent_id in range(runner.num_agents):
                action, _, rnn_state = runner.actor[agent_id].get_actions(
                    obs[:, agent_id, :], rnn_states[:, agent_id, :], masks[:, agent_id, :], None, None
                )
                action_space = action.shape[1]
                actions[:, agent_id, :action_space] = action
                rnn_states[:, agent_id, :] = rnn_state

            # runner.env.unwrapped._commands[:, :] = move_vector

            # runner.env.unwrapped.command_manager._terms['base_velocity'].command[:, :] = torch.tensor(movements["rotate_left"]+movements["move_forward"],device="cuda:0")

            obs, _, rewards, dones, _, _ = runner.env.step(actions)

            total_rewards += rewards
            print(f"Average reward: {rewards.mean(axis=0)}")
            dones_env = torch.all(dones, axis=1)
            masks = torch.ones((args["num_envs"], runner.num_agents, 1), dtype=torch.float64, device="cuda:0")
            masks[dones_env] = torch.zeros(
                ((dones_env).sum(), runner.num_agents, 1), dtype=torch.float64, device="cuda:0"
            )
            rnn_states[dones_env] = torch.zeros(
                ((dones_env).sum(), runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
                dtype=torch.float64,
                device="cuda:0",
            )

    runner.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
