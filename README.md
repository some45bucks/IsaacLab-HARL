# For Heterogeneous Agent Reinforcement Learning

<!-- ![GIF 1](withoutpolicy.gif) ![GIF 2](withpolicy.gif) -->

<div style="display: flex; justify-content: space-around;">
    <img src="withoutpolicy.gif" width="45%">
    <img src="withpolicy.gif" width="45%">
</div>

*Before and after results of multi-agent reinforcement learning for bar balancing task*

---
Use [these instructions](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html#install-isaac-sim-using-pip) to install isaac sim.

Make sure to run all the following commands with your isaaclab conda environment activated, as highlighted in the instructions provided above.


```bash
cd IsaacLab
./isaaclab.sh -i
pip install gymnasium==0.29.0
```
Note that we modified the `isaaclab.sh` install script to point to our fork of the [HARL repository](https://github.com/some45bucks/HARL.git), which the script will automatically install into your conda environment. If you want to make changes to the HARL code yourself, with your conda environment activated run the following

```bash
git clone https://github.com/some45bucks/HARL.git
cd HARL
pip install -e .
```
This will uninstall the default HARL and reference this one instead, recognizing any changes to the code that you make.


At this point you should be able to run the trained cooperative bar balancing task.

```bash
cd IsaacLab
python source/standalone/workflows/harl/play_keyboard.py --task Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0 --num_envs 1 --algorithm happo --dir source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/anymal_c_multi_agent/best_model
```

Which will allow you to interact with the trained by using the keyboard to give velocity commands.

```
a,d = angular velocity commands (left, right)
arrow keys = x,y velocity commands (forward, backward, left, right)
```

If you are just interested in running the environment and seeing how the environment resets, training works etc. You can use the `play.py` script instead.

If you want to use the soccer or piano environments, unzip the assets.zip folder into the Isaaclab directory.


# Training custom environment


```bash
python source/standalone/workflows/harl/train.py  --task Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0 --video_interval 10_000 --num_envs 2048 --save_interval 10 --log_interval 10 --algorithm happo --headless --num_env_steps 2_000_000_000
```

You can run `python play.py -h` to see all arguments. Also note that the configuration for the algorithms can be found in their yaml files, an
example of which can be found at `source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/anymal_c_multi_agent/agents/harl_ppo_cfg.yaml`, however, these will be overwritten with any parameters you define in the `train.py` script.


# Registering a New Environment in Isaac Lab

Isaac Lab allows you to register custom environments in the Gym framework, enabling efficient use with Gym's `make` command. This guide provides steps to register a new environment, using the example of the `DirectEnvRL`-based Piano Movers environment.

---

## Environment Registration Example

The best way to understand how to set up your own custom environment is by looking at the environment set up at `IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/anymal_c_multi_agent`, which is the bar balancing environment. A simple way to get started is to make a copy of this folder and begin modifying the environment to your specified needs. This folder contains the environment code and the necessary registration logic. To register your custom environment, follow these steps:

---

## Steps to Register a New Environment

### 1. **Modify Extensions**

If you are adding a new `DirectEnvRL` environment, place your code in the following directory:

```
IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/
```

### 2. **Update `__init__.py`**

Ensure that your new environment is registered in the Gym framework by adding the appropriate registration logic to the `__init__.py` file in the corresponding directory.

Now you can use the instructions provided at the beginning of this README for training the environment. We advise you also utilize the debugging tools provided in Visual Studio Code and setup the corresponding `launch.json` file to aid in your development.


![Isaac Lab](docs/source/_static/isaaclab.jpg)
# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows, such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html), it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks. Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.

## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:
- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
