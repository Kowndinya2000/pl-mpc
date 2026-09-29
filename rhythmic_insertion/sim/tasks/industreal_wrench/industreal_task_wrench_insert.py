# Copyright (c) 2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Wrench–nut insertion dynamics, observations, rewards and resets."""

import hydra
import numpy as np
import warp as wp
import omegaconf
import os
import torch

from isaacgym import gymapi, gymtorch, torch_utils
from rhythmic_insertion.sim.tasks.industreal_wrench.schema.factory_schema_class_task import FactoryABCTask
from rhythmic_insertion.sim.tasks.industreal_wrench.schema.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
import rhythmic_insertion.sim.tasks.industreal_wrench.industreal_algo_utils as algo_utils
from rhythmic_insertion.sim.tasks.industreal_wrench.industreal_env_wrench import IndustRealEnvWrench
from rhythmic_insertion.sim.utils import torch_jit_utils

class IndustRealTaskWrenchInsert(IndustRealEnvWrench, FactoryABCTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        """Initialize instance variables. Initialize task superclass."""

        self.cfg = cfg
        self._get_task_yaml_params()

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        self._acquire_task_tensors()
        self.parse_controller_spec()

        # Get Warp mesh objects for SAPU and SDF-based reward (sdf mode only)
        if self.cfg_task.rl.reward_type == "sdf":
            wp.init()
            self.wp_device = wp.get_device(self.device)
            (
                self.wp_plug_meshes,
                self.wp_plug_meshes_sampled_points,
                self.wp_socket_meshes,
            ) = algo_utils.load_asset_meshes_in_warp(
                plug_files=self.plug_files,
                socket_files=self.socket_files,
                num_samples=self.cfg_task.rl.sdf_reward_num_samples,
                device=self.wp_device,
            )

        if self.viewer != None:
            self._set_viewer_params()

        self._load_socket_init_pose_options()
        self._init_state_cache()

    def _load_socket_init_pose_options(self):
        asset_root = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    "..", "..", "..", "assets"
                )
            )

        socket_init_pose_options_list = []
        z_positions_list = []
        z_min_list = []
        z_max_list = []
        for subassembly in self.cfg_env.env.desired_subassemblies:
            poses = torch.from_numpy(
                    np.load(
                        os.path.join(asset_root, f"screw/initial_poses/{subassembly}/filtered.npy")
                    )
                ).to(self.device) # (num_options, 7)
            socket_init_pose_options_list.append(poses)
            z_positions_list.append(poses[:, 2])
            z_min_list.append(torch.min(poses[:, 2]))
            z_max_list.append(torch.max(poses[:, 2]))
        
        # Find the minimum and maximum Z values from all options
        z_min = torch.stack(z_min_list) # (num_subassemblies,)
        z_max = torch.stack(z_max_list) # (num_subassemblies,)

        # Extract the noise ratio bounds
        min_ratio, max_ratio = self.cfg_task.randomize.socket_pos_z_noise_ratio_bounds
        min_ratio = np.clip(min_ratio, 0., 1.)
        max_ratio = np.clip(max_ratio, 0., 1.)
        if min_ratio > max_ratio:
            max_ratio = min_ratio

        # Calculate the lower and upper Z bounds based on the ratios
        z_lower_bound = z_min + (z_max - z_min) * min_ratio # (num_subassemblies,)
        z_upper_bound = z_min + (z_max - z_min) * max_ratio # (num_subassemblies,)

        # Filter the pose options based on the Z bounds
        self.socket_init_pose_options = []
        for i, subassembly in enumerate(self.cfg_env.env.desired_subassemblies):
            if z_lower_bound[i] == z_upper_bound[i]:
                options_i = socket_init_pose_options_list[i][(z_positions_list[i] >= z_lower_bound[i])][:1]
            else:
                options_i = socket_init_pose_options_list[i][(z_positions_list[i] >= z_lower_bound[i]) & (z_positions_list[i] <= z_upper_bound[i])]

            self.socket_init_pose_options.append(
                options_i
            )
            print(f"    {subassembly}: {len(self.socket_init_pose_options[-1])} valid poses")

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        # Derive numObservations from observation_mode before VecTask reads it.
        _obs_mode_sizes = {"full": 17, "minimal": 7}
        obs_mode = self.cfg_task.rl.observation_mode
        assert obs_mode in _obs_mode_sizes, (
            f"Unknown observation_mode '{obs_mode}'. Valid options: {list(_obs_mode_sizes)}"
        )
        self.cfg["env"]["numObservations"] = _obs_mode_sizes[obs_mode]

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # Compute pose of wrench head goal and top of socket in socket frame
        self.wrench_head_goal_pos_local = torch.tensor(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                ]
                for i in range(self.num_envs)
            ],
            device=self.device,
        )
        roll = torch.zeros((self.num_envs, ), device=self.device)
        pitch = -90/180 * torch.pi * torch.ones((self.num_envs, ), device=self.device)
        yaw = -90/180 * torch.pi * torch.ones((self.num_envs, ), device=self.device)
        self.wrench_head_goal_quat_local = torch_jit_utils.quat_from_euler_xyz(roll, pitch, yaw)

        self.socket_top_pos_local = torch.tensor(
            [[0.0, 0.0, self.socket_heights[i]/2] for i in range(self.num_envs)],
            device=self.device,
        )
        self.socket_quat_local = self.identity_quat.clone()

        self.standard_socket_quat_local = None

        self.keypoint_offsets_plug = (
            algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device, axis=0)
            * self.cfg_task.rl.keypoint_scale
        )
        self.keypoint_offsets_socket = (
            algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device, axis=-1)
            * self.cfg_task.rl.keypoint_scale
        )

        self.keypoints_plug = torch.zeros(
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.keypoints_socket = torch.zeros_like(
            self.keypoints_plug, device=self.device
        )

        self.actions = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )

        self.curr_max_disp = self.cfg_task.rl.initial_max_disp

        ### Initialize tensors for plug resetting
        self.curriculum_disp = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.plug_pos_xy_noise = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.plug_rot_noise = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        if self.standard_socket_quat_local is not None:
            self.standard_socket_quat = torch_jit_utils.quat_mul(self.socket_quat, self.standard_socket_quat_local)
        else:
            # Get the transformation from socket pose to standard socket pose, according to the current socket pose
            standard_socket_quat = self.calc_feasible_nut_quaternions(self.socket_quat.clone())[:, -4:] # (N, 4)
            self.standard_socket_quat_local = torch_jit_utils.quat_mul(torch_jit_utils.quat_conjugate(self.socket_quat), standard_socket_quat)
            self.standard_socket_quat = standard_socket_quat.clone() # (N, 4)
        
        # Compute pose of gripper goal and top of socket in global frame
        self.wrench_head_goal_quat, self.wrench_head_goal_pos = torch_jit_utils.tf_combine(
            
            self.standard_socket_quat,
            self.socket_pos,
            self.wrench_head_goal_quat_local,
            self.wrench_head_goal_pos_local,
        )
        self.plug_goal_quat, self.plug_goal_pos = torch_jit_utils.tf_combine(
            self.wrench_head_goal_quat, 
            self.wrench_head_goal_pos,
            self.plug_to_wrench_head_quat, 
            self.plug_to_wrench_head_pos,
        )
        self.gripper_goal_quat, self.gripper_goal_pos = torch_jit_utils.tf_combine(
            self.wrench_head_goal_quat, 
            self.wrench_head_goal_pos,
            self.fingertip_midpoint_to_wrench_head_quat, 
            self.fingertip_midpoint_to_wrench_head_pos,
        )

        self.socket_top_quat, self.socket_top_pos = torch_jit_utils.tf_combine(
            
            self.standard_socket_quat,
            self.socket_pos,
            self.socket_quat_local,
            self.socket_top_pos_local,
        )

        # Add observation noise to socket pos
        self.noisy_socket_pos = torch.zeros_like(
            self.socket_pos, dtype=torch.float32, device=self.device
        )
        socket_obs_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

        # Add observation noise to socket rot
        socket_rot_euler = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        roll, pitch, yaw = torch_jit_utils.get_euler_xyz(self.standard_socket_quat)
        socket_rot_euler[:, 0] = roll
        socket_rot_euler[:, 1] = pitch
        socket_rot_euler[:, 2] = yaw

        socket_obs_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
        self.noisy_socket_quat = torch_jit_utils.quat_from_euler_xyz(
            socket_obs_rot_euler[:, 0],
            socket_obs_rot_euler[:, 1],
            socket_obs_rot_euler[:, 2],
        )

        # Compute observation noise on socket
        (
            self.noisy_wrench_head_goal_quat, 
            self.noisy_wrench_head_goal_pos,
        ) = torch_jit_utils.tf_combine(
            self.noisy_socket_quat,
            self.noisy_socket_pos,
            self.wrench_head_goal_quat_local,
            self.wrench_head_goal_pos_local,
        )
        (
            self.noisy_gripper_goal_quat, 
            self.noisy_gripper_goal_pos,
        ) = torch_jit_utils.tf_combine(
            self.noisy_wrench_head_goal_quat, 
            self.noisy_wrench_head_goal_pos,
            self.fingertip_midpoint_to_wrench_head_quat, 
            self.fingertip_midpoint_to_wrench_head_pos,
        )

        # Compute pos of keypoints on plug and socket in world frame
        for idx, (keypoint_offset_plug, keypoint_offset_socket) in enumerate(zip(self.keypoint_offsets_plug, self.keypoint_offsets_socket)):
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(
                self.wrench_head_quat,
                self.wrench_head_pos,
                self.identity_quat,
                keypoint_offset_plug.repeat(self.num_envs, 1),
            )[1]

            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(
                
                self.standard_socket_quat,
                self.socket_pos,
                self.identity_quat,
                keypoint_offset_socket.repeat(self.num_envs, 1),
            )[1]

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(
            self.device
        )  

        self._apply_object_actions_as_ctrl_targets(
            actions=self.actions, 
            ctrl_target_gripper_dof_pos=self.gripper_dof_pos_close.clone(), 
            do_scale=True,
        )

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        ## convert poses to socket frame
        # calc world to socket transform
        socket_quat_inv, socket_pos_inv = torch_jit_utils.tf_inverse(self.standard_socket_quat, self.socket_pos)
        noisy_socket_quat_inv, noisy_socket_pos_inv = torch_jit_utils.tf_inverse(self.noisy_socket_quat, self.noisy_socket_pos)
        # wrench head
        wrench_head_quat_socket, wrench_head_pos_socket = torch_jit_utils.tf_combine(
            socket_quat_inv,
            socket_pos_inv,
            self.wrench_head_quat,
            self.wrench_head_pos,
        )
        noisy_wrench_head_quat_socket, noisy_wrench_head_pos_socket = torch_jit_utils.tf_combine(
            noisy_socket_quat_inv,
            noisy_socket_pos_inv,
            self.wrench_head_quat,
            self.wrench_head_pos,
        )
        # plug
        plug_quat_socket, plug_pos_socket = torch_jit_utils.tf_combine(
            socket_quat_inv,
            socket_pos_inv,
            self.plug_quat,
            self.plug_pos,
        )
        ## delta pos
        delta_pos = wrench_head_pos_socket - self.wrench_head_goal_pos_local
        noisy_delta_pos = noisy_wrench_head_pos_socket - self.wrench_head_goal_pos_local

        ## Define observations (for actor)
        if self.cfg_task.rl.observation_mode == "minimal":
            obs_tensors = [
                noisy_wrench_head_pos_socket,   # 3
                noisy_wrench_head_quat_socket,  # 4
            ]  # 7
        else:  # "full"
            obs_tensors = [
                noisy_wrench_head_pos_socket,   # 3
                noisy_wrench_head_quat_socket,  # 4

                self.wrench_head_goal_pos_local,  # 3
                self.wrench_head_goal_quat_local,  # 4

                noisy_delta_pos,  # 3
            ]  # 17

        # Define state (for critic)
        state_tensors = [
            self.arm_dof_pos,  # 7
            self.arm_dof_vel,  # 7

            wrench_head_pos_socket,  # 3
            wrench_head_quat_socket,  # 4

            self.fingertip_centered_linvel,  # 3
            self.fingertip_centered_angvel,  # 3

            self.wrench_head_goal_pos_local,  # 3
            self.wrench_head_goal_quat_local,  # 4

            delta_pos,  # 3

            plug_pos_socket,  # 3
            plug_quat_socket,  # 4

            noisy_delta_pos - delta_pos,  # 3
        ]  # 47

        self.obs_buf = torch.cat(
            obs_tensors, dim=-1
        )  
        self.states_buf = torch.cat(state_tensors, dim=-1)

        return self.obs_buf

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        ## Backup Previous Reward
        self.prev_rew_buf = self.rew_buf.clone()

        ## Check Current State
        is_plug_engaged_w_socket = algo_utils.check_plug_engaged_w_socket(
                
                plug_pos=self.wrench_head_pos,
                socket_pos=self.socket_pos,
                keypoints_plug=self.keypoints_plug,
                keypoints_socket=self.keypoints_socket,
                socket_heights=self.socket_heights,
                plug_thicknesses=self.plug_thicknesses,
                cfg_task=self.cfg_task,
                progress_buf=self.progress_buf,
            )
        is_plug_inserted_in_socket = algo_utils.check_plug_inserted_in_socket(
            plug_pos=self.wrench_head_pos,
            socket_pos=self.socket_pos,
            keypoints_plug=self.keypoints_plug,
            keypoints_socket=self.keypoints_socket,
            socket_heights=self.socket_heights,
            cfg_task=self.cfg_task,
            progress_buf=self.progress_buf,
        )

        ## Calculate Reward
        if self.cfg_task.rl.reward_type == "sdf":
            # SDF-Based Reward
            sdf_reward = algo_utils.get_sdf_reward(
                wp_plug_meshes_sampled_points=self.wp_plug_meshes_sampled_points,
                asset_indices=self.asset_indices,
                plug_pos=self.plug_pos,
                plug_quat=self.plug_quat,
                plug_goal_sdfs=self.plug_goal_sdfs,
                wp_device=self.wp_device,
                device=self.device,
            )
            # Engagement bonus scaled by height closeness to full insertion
            engagement_reward_scale = algo_utils.get_engagement_reward_scale(
                plug_pos=self.wrench_head_pos,
                socket_pos=self.socket_pos,
                is_plug_engaged_w_socket=is_plug_engaged_w_socket,
                success_height_thresh=self.cfg_task.rl.success_height_thresh,
                device=self.device,
            )
            # SAPU: reward scale based on interpenetration distance
            low_interpen_envs, high_interpen_envs = [], []
            (
                low_interpen_envs,
                high_interpen_envs,
                sapu_reward_scale,
            ) = algo_utils.get_sapu_reward_scale(
                asset_indices=self.asset_indices,
                plug_pos=self.plug_pos,
                plug_quat=self.plug_quat,
                socket_pos=self.socket_pos,
                socket_quat=self.socket_quat,
                wp_plug_meshes_sampled_points=self.wp_plug_meshes_sampled_points,
                wp_socket_meshes=self.wp_socket_meshes,
                interpen_thresh=self.cfg_task.rl.interpen_thresh,
                wp_device=self.wp_device,
                device=self.device,
            )
            # Filter high-interpenetration envs from success/engagement
            if self.cfg_task.rl.filter_high_interpen and len(high_interpen_envs) > 0:
                is_plug_inserted_in_socket[high_interpen_envs] = False
                is_plug_engaged_w_socket[high_interpen_envs] = False
            # Combine: SDF + engagement, then SAPU
            self.rew_buf[:] = self.cfg_task.rl.sdf_reward_scale * sdf_reward
            self.extras["sdf_reward"] = torch.mean(self.rew_buf)
            self.rew_buf[:] += engagement_reward_scale * self.cfg_task.rl.engagement_bonus
            self.extras["sdf_plus_engagement_reward"] = torch.mean(self.rew_buf)
            self.rew_buf[low_interpen_envs] *= sapu_reward_scale
            if len(high_interpen_envs) > 0:
                self.rew_buf[high_interpen_envs] = self.prev_rew_buf[high_interpen_envs]
            self.extras["sapu_adjusted_reward"] = torch.mean(self.rew_buf)
            self.extras["low_interpen_percentage"] = low_interpen_envs.shape[0] / self.num_envs
        else:  # binary
            self.rew_buf[:] = is_plug_inserted_in_socket.float()

        ## record the first step of success and engagement
        new_success_envs = torch.nonzero(
            is_plug_inserted_in_socket & ~self.success_buf.to(torch.bool)
            ).flatten()
        if len(new_success_envs) > 0:
            self.success_step_buf[new_success_envs] = self.progress_buf[new_success_envs]
        new_engagement_envs = torch.nonzero(
            is_plug_engaged_w_socket & ~self.engagement_buf.to(torch.bool)
            ).flatten()
        if len(new_engagement_envs) > 0:
            self.engagement_step_buf[new_engagement_envs] = self.progress_buf[new_engagement_envs]

        ## update accumulated success and engagement
        self.success_buf[is_plug_inserted_in_socket] = 1
        self.engagement_buf[is_plug_engaged_w_socket] = 1
        
        ## Visualize the success and engagement
        cube_poses = torch.cat([self.socket_pos, self.standard_socket_quat], dim=-1)[:, None, :]
        cube_colors = torch.zeros((self.num_envs, 3), device=self.device)
        cube_colors[is_plug_engaged_w_socket] = torch.tensor([255., 0., 0.], device=self.device)
        cube_colors[is_plug_inserted_in_socket] = torch.tensor([0., 255., 0.], device=self.device)
        self.draw_cubes(cube_poses, cube_colors, line_length=0.2)

        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        if is_last_step:

            # Log average steps to success and engagement
            num_successes = torch.sum(self.success_buf)
            if num_successes > 0:
                self.extras["avg_steps_to_success"] = torch.mean(
                    self.success_step_buf[self.success_buf.bool()].float()
                )
                self.extras["std_steps_to_success"] = torch.std(
                    self.success_step_buf[self.success_buf.bool()].float()
                ) if num_successes > 1 else torch.tensor(self.cfg_task.rl.max_episode_length, dtype=torch.float32, device=self.device)
            else:
                self.extras["avg_steps_to_success"] = torch.tensor(self.cfg_task.rl.max_episode_length, dtype=torch.float32, device=self.device)
                self.extras["std_steps_to_success"] = torch.tensor(self.cfg_task.rl.max_episode_length, dtype=torch.float32, device=self.device)
            num_engagements = torch.sum(self.engagement_buf)
            if num_engagements > 0:
                self.extras["avg_steps_to_engagement"] = torch.mean(
                    self.engagement_step_buf[self.engagement_buf.bool()].float()
                )
                self.extras["std_steps_to_engagement"] = torch.std(
                    self.engagement_step_buf[self.engagement_buf.bool()].float()
                ) if num_engagements > 1 else torch.tensor(self.cfg_task.rl.max_episode_length, dtype=torch.float32, device=self.device)
            else:
                self.extras["avg_steps_to_engagement"] = torch.tensor(self.cfg_task.rl.max_episode_length, dtype=torch.float32, device=self.device)
                self.extras["std_steps_to_engagement"] = torch.tensor(self.cfg_task.rl.max_episode_length, dtype=torch.float32, device=self.device)

            # Success: Log success rate of the episode
            self.extras["insertion_successes"] = torch.mean(
                self.success_buf.float()
            )
            self.extras["insertion_engagements"] = torch.mean(
                self.engagement_buf.float()
            )

            # Final Success: Log success rate of the last step
            self.extras["last_step_insertion_successes"] = is_plug_inserted_in_socket.float().mean()
            self.extras["last_step_insertion_engagements"] = is_plug_engaged_w_socket.float().mean()

            # Per-env steps to success: step index when first succeeded, -1 if never succeeded.
            self.extras["steps_to_success"] = torch.where(
                self.success_buf.bool(),
                self.success_step_buf,
                torch.full_like(self.success_step_buf, -1),
            )

            # SBC: scale reward by curriculum difficulty, then update difficulty
            sbc_rew_scale = algo_utils.get_curriculum_reward_scale(
                cfg_task=self.cfg_task, curr_max_disp=self.curr_max_disp
            )
            self.rew_buf[:] = torch.where(
                self.rew_buf[:] < 0.0,
                self.rew_buf[:] / sbc_rew_scale,
                self.rew_buf[:] * sbc_rew_scale,
            )
            self.curr_max_disp = algo_utils.get_new_max_disp(
                curr_success=self.extras["insertion_successes"],
                cfg_task=self.cfg_task,
                curr_max_disp=self.curr_max_disp,
            )

            self.extras["curr_max_disp"] = self.curr_max_disp

            ## Statistics

    def _update_reset_buf(self):
        """Assign environments for reset if maximum episode length has been reached."""

        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self._state_bank is not None:
            self._restore_cached_state()
            if self.cfg_task.rl.reward_type == "sdf":
                self.plug_goal_sdfs = algo_utils.get_plug_goal_sdfs(
                    wp_plug_meshes=self.wp_plug_meshes,
                    asset_indices=self.asset_indices,
                    socket_pos=self.plug_goal_pos,
                    socket_quat=self.plug_goal_quat,
                    wp_device=self.wp_device,
                )
            self._reset_buffers()
            return

        self._reset_kuka()

        # Close gripper onto plug
        self.disable_gravity()  # to prevent plug from falling
        self._reset_object()

        self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)

        self.init_wrench_head_pos, self.init_wrench_head_quat = None, None

        self._move_gripper_to_init_plug_pose(
            sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
        )      
        self.enable_gravity()

        if self.cfg_task.rl.reward_type == "sdf":
            self.plug_goal_sdfs = algo_utils.get_plug_goal_sdfs(
                wp_plug_meshes=self.wp_plug_meshes,
                asset_indices=self.asset_indices,
                socket_pos=self.plug_goal_pos,
                socket_quat=self.plug_goal_quat,
                wp_device=self.wp_device,
            )

        self._reset_buffers()

        ## record the poses

    # ──────────────────────────────────────────────────────────────────────────
    # State caching helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_cache_path(self, path):
        """Resolve a relative cache_states_path against the repo root."""
        if path is None or os.path.isabs(path):
            return path
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        return os.path.join(repo_root, path)

    def _init_state_cache(self):
        """Load the configured initial-state bank from disk."""
        cache_path = self._resolve_cache_path(
            getattr(self.cfg_task.env, "cache_states_path", None)
        )
        self._state_bank = None
        if cache_path is not None and os.path.isfile(cache_path):
            with np.load(cache_path, allow_pickle=False) as archive:
                names = archive["subassemblies"].tolist()
                if names != list(self.cfg_env.env.desired_subassemblies):
                    raise ValueError("Reset-state asset sizes do not match the configured subassemblies.")
                bank = {k: torch.as_tensor(archive[k], device=self.device)
                        for k in archive.files if k != "subassemblies"}
            # Verify every requested asset_index has at least one saved row.
            for ai in set(self.asset_indices):
                n = (bank["asset_index"] == ai).sum().item()
                if n == 0:
                    raise RuntimeError(
                        f"State cache '{cache_path}' has no rows for asset_index={ai}. "
                        "Re-generate the cache with matching desired_subassemblies."
                    )
            self._state_bank = bank

    def _restore_cached_state(self):
        """Sample one snapshot per env from the bank (matching asset_index) and apply it."""
        bank = self._state_bank
        bank_ai = bank["asset_index"]  # (N,)
        asset_index_tensor = torch.tensor(self.asset_indices, dtype=torch.long, device=self.device)

        # Build a (num_envs,) index into the bank, respecting asset_index.
        selected = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        for ai in torch.unique(asset_index_tensor).tolist():
            env_mask = asset_index_tensor == ai
            row_ids = (bank_ai == ai).nonzero(as_tuple=True)[0]  # global bank indices
            rand = torch.randint(0, len(row_ids), (int(env_mask.sum().item()),), device=self.device)
            selected[env_mask] = row_ids[rand]

        # Restore DOF positions; zero velocities and torques for stability.
        self.dof_pos[:] = bank["dof_pos"][selected].to(self.device)
        self.dof_vel[:] = 0.0
        self.dof_torque[:] = 0.0
        kuka_ids = self.kuka_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(kuka_ids),
            len(kuka_ids),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_torque),
            gymtorch.unwrap_tensor(kuka_ids),
            len(kuka_ids),
        )

        # Restore object poses; zero all object velocities for stability.
        self.root_state[self.plug_actor_ids_sim, :7] = bank["plug_root"][selected, :7].to(self.device)
        self.root_state[self.socket_actor_ids_sim, :7] = bank["socket_root"][selected, :7].to(self.device)
        self.root_state[self.bolt_actor_ids_sim, :7] = bank["bolt_root"][selected, :7].to(self.device)
        self.root_state[self.plug_actor_ids_sim, 7:] = 0.0
        self.root_state[self.socket_actor_ids_sim, 7:] = 0.0
        self.root_state[self.bolt_actor_ids_sim, 7:] = 0.0
        obj_ids = torch.cat(
            [self.plug_actor_ids_sim, self.socket_actor_ids_sim, self.bolt_actor_ids_sim]
        ).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(obj_ids),
            len(obj_ids),
        )

        # Restore controller targets.
        self.ctrl_target_dof_pos[:] = bank["ctrl_target_dof_pos"][selected].to(self.device)
        self.ctrl_target_fingertip_centered_pos[:] = bank["ctrl_target_fingertip_pos"][selected].to(self.device)
        self.ctrl_target_fingertip_centered_quat[:] = bank["ctrl_target_fingertip_quat"][selected].to(self.device)

        # Force recompute of socket-derived cached tensor (depends on socket orientation).
        self.standard_socket_quat_local = None

        self.simulate_and_refresh()

    # ──────────────────────────────────────────────────────────────────────────

    def _reset_kuka(self):
        """Reset DOF states, DOF torques, and DOF targets of Kuka."""

        # Randomize DOF pos
        self.dof_pos[:, :7] = torch.tensor(self.cfg_task.randomize.kuka_arm_initial_dof_pos, device=self.device).repeat((self.num_envs, 1))
        self.dof_pos[:, 7:] = self.gripper_dof_pos_open.clone()
        
        # Stabilize Kuka
        self.dof_vel[:, :] = 0.0  
        self.dof_torque[:, :] = 0.0

        # Set DOF state
        kuka_actor_ids_sim = self.kuka_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(kuka_actor_ids_sim),
            len(kuka_actor_ids_sim),
        )

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_torque),
            gymtorch.unwrap_tensor(kuka_actor_ids_sim),
            len(kuka_actor_ids_sim),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

        self.ctrl_target_dof_pos = self.dof_pos.clone()
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()

    def _reset_object(self):
        """Reset root state of plug and socket."""
        ### Resetting the plug(wrench) first can avoid collision
        self._reset_plug()
        self._reset_socket()
        ### Resetting the socket(nut) first causes collision and introduces randomness in relative initial poses of plug and socket

    def _reset_socket(self):
        """Reset root state of socket."""
        
        ## BOLT ##
        self.bolt_pos[:, :] = torch.tensor(self.cfg_task.randomize.socket_base_pos_initial, device=self.device).repeat((self.num_envs, 1))
        self.bolt_pos[:, 0] += torch.empty(self.num_envs, device=self.device).uniform_(
            -self.cfg_task.randomize.socket_base_pos_noise[0], 
            self.cfg_task.randomize.socket_base_pos_noise[0]
        )
        self.bolt_pos[:, 1] += torch.empty(self.num_envs, device=self.device).uniform_(
            -self.cfg_task.randomize.socket_base_pos_noise[1], 
            self.cfg_task.randomize.socket_base_pos_noise[1]
        )
        self.bolt_pos[:, 2] += torch.empty(self.num_envs, device=self.device).uniform_(
            -self.cfg_task.randomize.socket_base_pos_noise[2], 
            self.cfg_task.randomize.socket_base_pos_noise[2]
        ) + self.cfg_base.env.table_height

        self.bolt_quat[:, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

        ## SOCKET (NUT) ##
        asset_indices = torch.tensor(self.asset_indices, device=self.device, dtype=torch.int32) # (num_envs, )
        self.socket_pos[:, :] = self.bolt_pos.clone()
        for j in range(len(self.cfg_env.env.desired_subassemblies)):
            subassembly_cnt = (asset_indices == j).sum()
            if subassembly_cnt > 0:
                sample_indices = torch.multinomial(torch.ones(self.socket_init_pose_options[j].shape[0]), subassembly_cnt, replacement=True)
                
                socket_pos_in_bolt = self.socket_init_pose_options[j][sample_indices, :3]
                self.socket_quat[asset_indices == j, :] = self.socket_init_pose_options[j][sample_indices, 3:7]
                self.socket_pos[asset_indices == j, :] += socket_pos_in_bolt
        
        # Stabilize socket
        self.socket_linvel[:, :] = 0.0
        self.socket_angvel[:, :] = 0.0

        # Set bolt and socket root state
        index_tensor = torch.concat([self.socket_actor_ids_sim, self.bolt_actor_ids_sim], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(index_tensor),
            len(index_tensor),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    def calc_feasible_nut_quaternions(self, nut_quat):
        r"""
        1. z-axis pointing up
        2. all six yaw rotations are feasible (60 degrees apart)
        3. sort the quaternions in the order of the positive yaw rotations, with the first yaw rotation closest to the world's yaw rotation
        ---
        nut_quat: (N, 4)
        ---
        return: (N, 6x4), [:, ix4: (i+1)x4] is the i-th feasible quaternion
        """
        N = nut_quat.shape[0]
        device = nut_quat.device
        dtype = nut_quat.dtype

        # z-axis pointing up
        nut_z_axis = torch_jit_utils.quat_axis(nut_quat, axis=2) # (N, 3)
        base_z_axis = torch.tensor([[0, 0, 1]], device=device, dtype=dtype).T # (3, 1)
        is_z_down = (nut_z_axis @ base_z_axis < 0).flatten() # (N, )

        inverting_z_quat = torch_jit_utils.quat_from_angle_axis(
            torch.tensor([np.pi], device=device, dtype=dtype).repeat(N), 
            torch.tensor([[1, 0, 0]], device=device, dtype=dtype).repeat(N, 1)
        ) # (N, 4)
        feasible_nut_quat = nut_quat.clone()
        feasible_nut_quat[is_z_down] = torch_jit_utils.quat_mul(nut_quat[is_z_down], inverting_z_quat[is_z_down]) # (N, 4)

        # all six yaw rotations are feasible (60 degrees apart)
        rotating_around_z_quat = torch_jit_utils.quat_from_angle_axis(
            torch.tensor([np.pi/3], device=device, dtype=dtype).repeat(N), 
            torch.tensor([[0, 0, 1]], device=device, dtype=dtype).repeat(N, 1)
        ) # (N, 4)

        world_x_axis = torch.tensor([1, 0, 0], device=device, dtype=dtype).unsqueeze(-1) # (3, 1)

        nut_quat_i = feasible_nut_quat.clone()
        nut_x_axis_i = torch_jit_utils.quat_axis(nut_quat_i, axis=0) # (N, 3)
        cos_dist_i = nut_x_axis_i @ world_x_axis # (N, 1)

        min_cos_dist = cos_dist_i # (N, 1)
        min_nut_quat = nut_quat_i # (N, 4)
        for i in range(5):
            nut_quat_i = torch_jit_utils.quat_mul(nut_quat_i, rotating_around_z_quat)
            nut_x_axis_i = torch_jit_utils.quat_axis(nut_quat_i, axis=0) # (N, 3)
            cos_dist_i = nut_x_axis_i @ world_x_axis # (N, 1)
            is_smaller = (cos_dist_i < min_cos_dist).squeeze(-1)
            min_cos_dist[is_smaller, :] = cos_dist_i[is_smaller, :]
            min_nut_quat[is_smaller, :] = nut_quat_i[is_smaller, :]

        feasible_nut_quaternions = torch.zeros((N, 6*4), device=device, dtype=dtype)
        feasible_nut_quaternions[:, 0:4] = min_nut_quat # feasible_nut_quat
        for i in range(5):
            feasible_nut_quaternions[:, (i+1)*4:(i+2)*4] = torch_jit_utils.quat_mul(feasible_nut_quaternions[:, i*4:(i+1)*4], rotating_around_z_quat)

        return feasible_nut_quaternions
    
    def _reset_plug(self):
        """Reset root state of plug to the grasp pose."""
        
        ## offset the plug to the fingertip center according to the plug/wrench thickness
        fingertip_to_palm_offset = 0.118 # 0.120457
        plug_to_fingertip_centered_pos = torch.zeros((self.num_envs, 3), device=self.device)
        plug_to_fingertip_centered_pos[:, 2] = -(fingertip_to_palm_offset - self.plug_thicknesses/2)

        plug_to_fingertip_centered_quat = torch_jit_utils.matrix_to_quaternion(torch.tensor([[
            [0.,-1., 0.],
            [-1., 0., 0.],
            [0., 0., -1.]
        ]], device=self.device)).repeat((self.num_envs, 1))[:, [1, 2, 3, 0]]

        init_plug_quat, init_plug_pos = torch_jit_utils.tf_combine(self.fingertip_centered_quat, self.fingertip_centered_pos, plug_to_fingertip_centered_quat, plug_to_fingertip_centered_pos)

        self.plug_pos[:, :] = init_plug_pos.clone()
        self.plug_quat[:, :] = init_plug_quat.clone()

        # Stabilize plug
        self.plug_linvel[:, :] = 0.0
        self.plug_angvel[:, :] = 0.0

        # Set plug root state
        plug_actor_ids_sim = self.plug_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(plug_actor_ids_sim),
            len(plug_actor_ids_sim),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    def _reset_buffers(self):
        """Reset buffers."""

        self.reset_buf[:] = 0
        self.progress_buf[:] = 0
        self.success_buf[:] = 0
        self.engagement_buf[:] = 0
        self.success_step_buf[:] = self.cfg_task.rl.max_episode_length
        self.engagement_step_buf[:] = self.cfg_task.rl.max_episode_length

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-0.5, 0.0, 0.8)
        cam_target = gymapi.Vec3(0.5, 0.0, 0.5)
        env_idx = min((int(np.sqrt(self.num_envs)) + self.num_envs) // 2, self.num_envs - 1)
        self.gym.viewer_camera_look_at(self.viewer, self.env_ptrs[env_idx], cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale
    ):
        """Apply fingertip actions from policy as position/rotation targets displacement."""

        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_centered_pos = (
            self.fingertip_centered_pos + pos_actions
        )

        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_jit_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_mul(
            rot_actions_quat, self.fingertip_centered_quat
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _apply_object_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale
    ):
        """Apply wrench head actions from policy as position/rotation targets displacement."""

        # Map to [-1, 1]
        actions = torch.clamp(actions, -1.0, 1.0)

        # Interpret actions as target pos displacements in the local frame and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )

        # Interpret actions as target rot (axis-angle) displacements in the local frame
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_jit_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )

        ctrl_target_wrench_head_quat, ctrl_target_wrench_head_pos = torch_jit_utils.tf_combine(
            self.wrench_head_quat, self.wrench_head_pos, rot_actions_quat, pos_actions)

        poses = torch.cat([
            torch.cat([self.wrench_head_pos, self.wrench_head_quat], dim=1)[:, None, :],
            torch.cat([ctrl_target_wrench_head_pos, ctrl_target_wrench_head_quat], dim=1)[:, None, :],
        ], dim=1)
        self.visualize_poses(poses)

        # Transform to gripper fingertip's action target
        self.ctrl_target_fingertip_centered_quat, self.ctrl_target_fingertip_centered_pos = torch_jit_utils.tf_combine(
            ctrl_target_wrench_head_quat, ctrl_target_wrench_head_pos,
            self.fingertip_midpoint_to_wrench_head_quat, self.fingertip_midpoint_to_wrench_head_pos
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        
        self.generate_ctrl_signals()
    
    def _get_init_wrench_head_pose(self, env_ids, resample: bool=True):

        if self.init_wrench_head_pos is None or self.init_wrench_head_quat is None:
            assert resample and torch.unique(env_ids).shape[0] == self.num_envs
            self.init_wrench_head_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.init_wrench_head_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

        if not resample:
            return self.init_wrench_head_pos[env_ids].clone(), self.init_wrench_head_quat[env_ids].clone()
        else:
            # Generate randomized downward displacement based on curriculum
            curr_curriculum_disp_range = (
                self.curr_max_disp - self.cfg_task.rl.curriculum_height_bound[0]
            )
            self.curriculum_disp[env_ids] = self.cfg_task.rl.curriculum_height_bound[0] + curr_curriculum_disp_range * (
                torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            )[env_ids]

            ### Generate plug pos & rotation randomness
            ## Position randomness
            # Sample planar points inside a unit square centered at the origin
            self.plug_pos_xy_noise[env_ids] = 2 * (
                torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
                - 0.5
            )[env_ids]
            self.plug_pos_xy_noise[env_ids] = self.plug_pos_xy_noise[env_ids] @ torch.diag(
                torch.tensor(
                    self.cfg_task.randomize.plug_pos_xy_noise,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            ## Rotation randomness
            self.plug_rot_noise[env_ids] = 2 * (
                torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
                - 0.5
            )[env_ids]
            self.plug_rot_noise[env_ids] = self.plug_rot_noise[env_ids] @ torch.diag(
                torch.tensor(
                    self.cfg_task.randomize.plug_rot_noise,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            angle = torch.norm(self.plug_rot_noise, p=2, dim=-1)
            axis = self.plug_rot_noise / angle.unsqueeze(-1)
            plug_rot_noise_quat = torch_jit_utils.quat_from_angle_axis(angle, axis)
            if self.cfg_task.rl.clamp_rot:
                plug_rot_noise_quat = torch.where(
                    angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                    plug_rot_noise_quat,
                    torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                        self.num_envs, 1
                    ),
                )

            # Get the transformation from socket pose to standard socket pose, according to the current socket pose
            # !!! Refresh the standard socket pose right before initializing the plug, to counter the (potential) rotation of the socket when initialized !!!
            standard_socket_quat = self.calc_feasible_nut_quaternions(self.socket_quat.clone())[:, -4:] # (N, 4)
            self.standard_socket_quat_local[env_ids] = torch_jit_utils.quat_mul(torch_jit_utils.quat_conjugate(self.socket_quat), standard_socket_quat)[env_ids]
            
            ## Assembled pose of the plug
            socket_pos = self.socket_pos.clone()
            socket_quat = standard_socket_quat.clone()

            # Set plug pos to assembled state, 
            wrench_head_quat, wrench_head_pos = torch_jit_utils.tf_combine(socket_quat, socket_pos, self.wrench_head_goal_quat_local, self.wrench_head_goal_pos_local)
            # offset so that plug's bottom is just touching the socket's top
            wrench_head_pos[:, 2] += (self.socket_heights + self.plug_thicknesses) / 2

            ## Curriculum: Apply curriculum displacement to plug
            # offset according to the curriculum displacement
            wrench_head_pos[:, 2] -= self.curriculum_disp
            
            ## Noises: Apply plug pos & rotation noise
            # Apply XY noise to plugs not partially inserted into sockets
            plug_not_partial_insert_idx = self.curriculum_disp < 0
            wrench_head_pos[plug_not_partial_insert_idx, :2] += self.plug_pos_xy_noise[plug_not_partial_insert_idx]
            # Apply rotation noise to plugs not partially inserted into sockets
            wrench_head_quat[plug_not_partial_insert_idx] = torch_jit_utils.quat_mul(
                wrench_head_quat[plug_not_partial_insert_idx], plug_rot_noise_quat[plug_not_partial_insert_idx]
            )

            self.init_wrench_head_pos[env_ids] = wrench_head_pos[env_ids].clone()
            self.init_wrench_head_quat[env_ids] = wrench_head_quat[env_ids].clone()
            
            return wrench_head_pos[env_ids].clone(), wrench_head_quat[env_ids].clone()

    def _move_gripper_to_init_plug_pose(self, sim_steps):
        """Move gripper to the initial pose of the plug."""

        ## Sample the initial pose of the wrench head
        wrench_head_pos, wrench_head_quat = self._get_init_wrench_head_pose(env_ids=torch.arange(self.num_envs), resample=True)

        ## move to higher pose
        target_wrench_head_quat, target_wrench_head_pos = wrench_head_quat.clone(), wrench_head_pos.clone()
        target_wrench_head_pos[:, 2] += 0.02

        self.ctrl_target_fingertip_centered_quat, self.ctrl_target_fingertip_centered_pos = torch_jit_utils.tf_combine(
            target_wrench_head_quat, target_wrench_head_pos, 
            self.fingertip_midpoint_to_wrench_head_quat, self.fingertip_midpoint_to_wrench_head_pos
        )
        
        self.move_gripper_to_target_pose(
            gripper_dof_pos=self.gripper_dof_pos_close,
            sim_steps=int(sim_steps*0.6),
        )

        ## move to the pose
        target_wrench_head_quat, target_wrench_head_pos = wrench_head_quat.clone(), wrench_head_pos.clone()

        self.ctrl_target_fingertip_centered_quat, self.ctrl_target_fingertip_centered_pos = torch_jit_utils.tf_combine(
            target_wrench_head_quat, target_wrench_head_pos, 
            self.fingertip_midpoint_to_wrench_head_quat, self.fingertip_midpoint_to_wrench_head_pos
        )
        
        self.move_gripper_to_target_pose(
            gripper_dof_pos=self.gripper_dof_pos_close,
            sim_steps=int(sim_steps*0.4),
        )

        # Stabilize Kuka
        self.dof_vel[:, :] = 0.0  
        self.dof_torque[:, :] = 0.0

        # Set DOF state
        kuka_actor_ids_sim = self.kuka_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(kuka_actor_ids_sim),
            len(kuka_actor_ids_sim),
        )

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_torque),
            gymtorch.unwrap_tensor(kuka_actor_ids_sim),
            len(kuka_actor_ids_sim),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

        self.ctrl_target_dof_pos = self.dof_pos.clone()
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()