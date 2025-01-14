# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBase, AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material_0 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass_0 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    add_base_mass_1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class AnymalCMultiAgentFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    # observation_space = 48
    observation_space = 235
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    robot_0.init_state.rot = (1.0, 0.0, 0.0, .75)
    robot_0.init_state.pos = (-1.0, 0.0, 1.0)


    robot_1: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_1/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    robot_1.init_state.rot = (1.0, 0.0, 0.0, .75)
    robot_1.init_state.pos = (1.0, 0.0, 1.0)

    # rec prism
    cfg_rec_prism= RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg( 
            size=(5,.1,.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, 2.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # we add a height scanner for perceptive locomotion
    height_scanner_0 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot_0/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_1 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot_1/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0


@configclass
class AnymalCMultiAgentRoughEnvCfg(AnymalCMultiAgentFlatEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner_0 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot_0/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_1 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot_1/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0


class AnymalCMultiAgent(DirectRLEnv):
    cfg: AnymalCMultiAgentFlatEnvCfg | AnymalCMultiAgentRoughEnvCfg

    def __init__(self, cfg: AnymalCMultiAgentFlatEnvCfg | AnymalCMultiAgentRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Joint position command (deviation from default joint positions)
        self.actions = torch.zeros(self.num_envs*self.num_robots, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.previous_actions = torch.zeros(
            self.num_envs*self.num_robots, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs*self.num_robots, 3, device=self.device)
        self._commands[:, 0] = 1.0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }

        self.base_ids = []
        self.feet_ids = []
        self.undesired_body_contact_ids = []

        for _contact_sensor in self.contact_sensors:
            _base_id, _ = _contact_sensor.find_bodies("base")
            _feet_ids, _ = _contact_sensor.find_bodies(".*FOOT")
            _undesired_contact_body_ids, _ = _contact_sensor.find_bodies(".*THIGH")
            self.base_ids.append(_base_id)
            self.feet_ids.append(_feet_ids)
            self.undesired_body_contact_ids.append(_undesired_contact_body_ids)

    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = []
        self.contact_sensors = []
        self.height_scanners = []
        self.object = RigidObject(self.cfg.cfg_rec_prism)
        
        self.scene.rigid_objects["object"] = self.object

        for i in range(self.num_robots):
            self.robots.append(Articulation(self.cfg.__dict__["robot_" + str(i)]))
            self.scene.articulations[f"robot_{i}"] = self.robots[i]
            self.contact_sensors.append(ContactSensor(self.cfg.__dict__["contact_sensor_" + str(i)]))
            self.scene.sensors[f"contact_sensor_{i}"] = self.contact_sensors[i]
            # if isinstance(self.cfg, AnymalCMultiAgentWalkingRoughEnvCfg):
                # we add a height scanner for perceptive locomotion
            self.height_scanners.append(RayCaster(self.cfg.__dict__["height_scanner_" + str(i)]))
            self.scene.sensors[f"height_scanner_{i}"] = self.height_scanners[i]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # We need to process the actions for each scene independently
        self.processed_actions = torch.zeros_like(actions)
        for i, robot in enumerate(self.robots):
            curr_robot = slice(self.num_envs * i, self.num_envs * (i + 1))
            self.actions[curr_robot] = actions[curr_robot].clone()
            self.processed_actions[curr_robot] = self.cfg.action_scale * self.actions[curr_robot] + robot.data.default_joint_pos

    def _apply_action(self):
        for i, robot in enumerate(self.robots):
            robot.set_joint_position_target(self.processed_actions[i])
        # self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self.previous_actions = self.actions.clone()
        height_datas = []
        for i in range(self.num_robots):
            height_data = None
            # if isinstance(self.cfg, AnymalCMultiAgentWalkingRoughEnvCfg):
            height_data = (
                self.height_scanners[i].data.pos_w[:, 2].unsqueeze(1) - self.height_scanners[i].data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
            height_datas.append(height_data)

        
        obs = []
        for i, robot in enumerate(self.robots):
            curr_robot = slice(self.num_envs * i, self.num_envs * (i + 1))
            obs.append(torch.cat(
            [
                tensor
                for tensor in (
                    robot.data.root_lin_vel_b,
                    robot.data.root_ang_vel_b,
                    robot.data.projected_gravity_b,
                    self._commands[curr_robot],
                    robot.data.joint_pos - robot.data.default_joint_pos,
                    robot.data.joint_vel,
                    height_datas[i],
                    self.actions[curr_robot],
                )
                if tensor is not None
            ],
            dim=-1,
            ))
        obs = torch.cat(obs, dim=0)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward = None
        for i, robot in enumerate(self.robots):
            curr_robot = slice(self.num_envs * i, self.num_envs * (i + 1))
            # linear velocity tracking
            lin_vel_error = torch.sum(torch.square(self._commands[curr_robot][:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1)
            lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
            # yaw rate tracking
            yaw_rate_error = torch.square(self._commands[curr_robot][:, 2] - robot.data.root_ang_vel_b[:, 2])
            yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
            # z velocity tracking
            z_vel_error = torch.square(robot.data.root_lin_vel_b[:, 2])
            # angular velocity x/y
            ang_vel_error = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)
            # joint torques
            joint_torques = torch.sum(torch.square(robot.data.applied_torque), dim=1)
            # joint acceleration
            joint_accel = torch.sum(torch.square(robot.data.joint_acc), dim=1)
            # action rate            
            action_rate = torch.sum(torch.square(self.actions[i] - self.previous_actions[i]).view(1,-1), dim=1)
            # feet air time
            first_contact = self.contact_sensors[i].compute_first_contact(self.step_dt)[:, self.feet_ids[i]]
            last_air_time = self.contact_sensors[i].data.last_air_time[:, self.feet_ids[i]]
            air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
                torch.norm(self._commands[curr_robot][:, :2], dim=1) > 0.1
            )
            # undersired contacts
            net_contact_forces = self.contact_sensors[i].data.net_forces_w_history
            is_contact = (
                torch.max(torch.norm(net_contact_forces[:, :, self.undesired_body_contact_ids[i]], dim=-1), dim=1)[0] > 1.0
            )
            contacts = torch.sum(is_contact, dim=1)
            # flat orientation
            flat_orientation = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)

            rewards = {
                "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
                "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
                "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
                "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
                "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
                "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
                "action_rate_l2": (action_rate * self.cfg.action_rate_reward_scale * self.step_dt).repeat(self.num_envs),
                "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
                "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
                "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            }
            curr_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            if reward is not None:
                reward += curr_reward
            else:
                reward = curr_reward

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     all_dones = []
    #     all_died = []
    #     for i in range(self.num_robots):
    #         time_out = self.episode_length_buf >= self.max_episode_length - 1
    #         net_contact_forces = self.contact_sensors[i].data.net_forces_w_history
    #         died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.base_ids[i]], dim=-1), dim=1)[0] > 1.0, dim=1)
    #         all_dones.append(time_out)
    #         all_died.append(died)
        
    #     return torch.any(torch.cat(all_dones), dim=0), torch.any(torch.cat(all_died), dim=0)

    #TODO: Implement a dones function that handles multiple robots 
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        contact_sensor = self.contact_sensors[0]
        base_id = self.base_ids[0]

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return False, False
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )
        self.object.write_root_state_to_sim(object_default_state, env_ids)
        # self.object.reset(env_ids)
        for robot in self.robots:
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = robot._ALL_INDICES
            robot.reset(env_ids)
            if len(env_ids) == self.num_envs:
                # Spread out the resets to avoid spikes in training when many environments reset at a similar time
                self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            self.actions[env_ids] = 0.0
            self.previous_actions[env_ids] = 0.0
            # Sample new commands
            self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
            # Reset robot state
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            # Logging
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.extras["log"] = dict()
            self.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
            self.extras["log"].update(extras)
