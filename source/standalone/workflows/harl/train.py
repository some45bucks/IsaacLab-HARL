"""Train an algorithm."""
import argparse
import sys
import json

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with HARL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
        "--algorihm",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c"
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c",
    )

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import random
from datetime import datetime
from harl.runners import ISSAC_LAB_RUNNER_REGISTRY

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

agent_cfg_entry_point = "harl_cfg_entry_point"

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):


    agent_cfg["eval"]["use_eval"] = False


    # #convert configs to dicts
    # args = vars(args_cli)
    # env_cfg_dict = env_cfg.to_dict()
    # env_cfg_dict['cfg_class'] = env_cfg

    # #remap args dict to match the expected args
    # args['env'] = args.pop('task')
    # args['exp_name'] = args['env']

    #create runner
    runner = ISSAC_LAB_RUNNER_REGISTRY[args_cli.algorithm](agent_cfg, env)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
