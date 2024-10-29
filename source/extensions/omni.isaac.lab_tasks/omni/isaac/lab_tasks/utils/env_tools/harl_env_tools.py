"""Tools for Issac Gym HARL."""
import gymnasium as gym
import os
import random
import numpy as np
import torch
from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from omni.isaac.lab_tasks.utils.wrappers.harl import HarlEnvWrapper


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape

# This will need to be modified to use in Isssac Lab
def make_train_env(env_name, seed, n_threads, env_cfg):

    env = gym.make(env_name, cfg=env_cfg, render_mode=None)

    env = HarlEnvWrapper(env)

    return env


def make_eval_env(env_name, seed, n_threads, env_args):
        raise NotImplementedError


def make_render_env(env_name, seed, env_cfg):
    env = gym.make(env_name, cfg=env_cfg, render_mode="rgb_array")

    # video_kwargs = {
    #     "video_folder": os.path.join(log_dir, "videos", "train"),
    #     "step_trigger": lambda step: step % args_cli.video_interval == 0,
    #     "video_length": args_cli.video_length,
    #     "disable_logger": True,
    # }
    # print("[INFO] Recording videos during training.")
    # print_dict(video_kwargs, nesting=4)
    # env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = HarlEnvWrapper(env)  # same as: `wrap_env(env, wrapper="auto")`
    
    return env, False, False, False, 1


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
        return envs.n_agents
