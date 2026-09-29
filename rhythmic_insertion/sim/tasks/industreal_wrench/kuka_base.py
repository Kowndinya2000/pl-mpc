# Copyright (c) 2021-2023, NVIDIA Corporation
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

"""KUKA iiwa and Robotiq simulation base and impedance controller."""

import hydra
import math
import numpy as np
import os
import sys
import torch

from gym import logger
from isaacgym import gymapi, gymtorch, torch_utils
from rhythmic_insertion.sim.tasks.base.vec_task import VecTask
import rhythmic_insertion.sim.tasks.industreal_wrench.factory_control as fc
from rhythmic_insertion.sim.tasks.industreal_wrench.schema.factory_schema_class_base import FactoryABCBase
from rhythmic_insertion.sim.tasks.industreal_wrench.schema.factory_schema_config_base import FactorySchemaConfigBase

# USE       : fingertip_centered_...: this is the fingertip written in URDF
# NOT USE   : fingertip_midpoint_...: this should be the fingertip calculated from the left, right, and middle finger(tips), but the calculation is not correct yet
## NOTE:
# no "plug" or "socket" in the code

class IndustRealKukaBase(VecTask, FactoryABCBase):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize VecTask superclass."""

        self.cfg = cfg
        self.cfg['headless'] = headless

        self._get_base_yaml_params()

        if self.cfg_base.mode.export_scene:
            sim_device = 'cpu'

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture,
                         force_render)  # create_sim() is called here

    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_base', node=FactorySchemaConfigBase)

        config_path = 'task/IndustRealKukaBase.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base['task']  # strip superfluous nesting

        asset_info_path = '../../assets/screw/yaml/screw_asset_info_kuka_table.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_kuka_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_kuka_table = self.asset_info_kuka_table['']['']['']['']['']['']['assets']['screw']['yaml']  # strip superfluous nesting

    def create_sim(self):
        """Set sim and PhysX params. Create sim object, ground plane, and envs."""

        if self.cfg_base.mode.export_scene:
            self.sim_params.use_gpu_pipeline = False

        self.sim = super().create_sim(compute_device=self.device_id,
                                      graphics_device=self.graphics_device_id,
                                      physics_engine=self.physics_engine,
                                      sim_params=self.sim_params)
        self._create_ground_plane()
        self.create_envs()  # defined in subclass

    def _create_ground_plane(self):
        """Set ground plane params. Add plane."""

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0  
        plane_params.static_friction = 1.0  
        plane_params.dynamic_friction = 1.0  
        plane_params.restitution = 0.0  

        self.gym.add_ground(self.sim, plane_params)

    def import_kuka_assets(self):
        """Set kuka and table asset options. Import assets."""

        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'kuka14_robotiq_description', 'urdf')
        kuka_file = 'kuka_robotiq_sdf.urdf'

        kuka_options = gymapi.AssetOptions()
        kuka_options.flip_visual_attachments = False
        kuka_options.fix_base_link = True
        kuka_options.collapse_fixed_joints = False
        kuka_options.thickness = 0.0  
        kuka_options.density = 1000.0  
        kuka_options.armature = 0.01  
        kuka_options.use_physx_armature = True
        if self.cfg_base.sim.add_damping:
            kuka_options.linear_damping = 1.0  
            kuka_options.max_linear_velocity = 1.0  
            kuka_options.angular_damping = 5.0  
            kuka_options.max_angular_velocity = 2 * math.pi  
        else:
            kuka_options.linear_damping = 0.0  
            kuka_options.max_linear_velocity = 1.0  
            kuka_options.angular_damping = 0.5  
            kuka_options.max_angular_velocity = 2 * math.pi  
        kuka_options.disable_gravity = True
        kuka_options.enable_gyroscopic_forces = True
        kuka_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE  # DOF_MODE_NONE
        kuka_options.use_mesh_materials = True
        if self.cfg_base.mode.export_scene:
            kuka_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        table_options = gymapi.AssetOptions()
        table_options.flip_visual_attachments = False  
        table_options.fix_base_link = True
        table_options.thickness = 0.0  
        table_options.density = 1000.0  
        table_options.armature = 0.0  
        table_options.use_physx_armature = True
        table_options.linear_damping = 0.0  
        table_options.max_linear_velocity = 1000.0  
        table_options.angular_damping = 0.0  
        table_options.max_angular_velocity = 64.0  
        table_options.disable_gravity = False
        table_options.enable_gyroscopic_forces = True
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            table_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        kuka_asset = self.gym.load_asset(self.sim, urdf_root, kuka_file, kuka_options)

        table_asset = self.gym.create_box(self.sim, self.asset_info_kuka_table.table_depth,
                                          self.asset_info_kuka_table.table_width, self.cfg_base.env.table_height,
                                          table_options)

        return kuka_asset, table_asset

    def acquire_base_tensors(self):
        """Acquire and wrap tensors. Create views."""

        '''
        The buffer has shape (num_actors, 13).
        State for each actor root contains:
        position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        '''

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  
        _body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)  
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)  
        _contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)  
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, 'kuka')  
        _mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, 'kuka')  
        _ft_sensors = self.gym.acquire_force_sensor_tensor(self.sim)

        self.ft_sensors = gymtorch.wrap_tensor(_ft_sensors)
        self.root_state = gymtorch.wrap_tensor(_root_state)
        self.body_state = gymtorch.wrap_tensor(_body_state)
        self.dof_state = gymtorch.wrap_tensor(_dof_state)
        self.dof_force = gymtorch.wrap_tensor(_dof_force)
        self.contact_force = gymtorch.wrap_tensor(_contact_force)
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        self.mass_matrix = gymtorch.wrap_tensor(_mass_matrix)

        self.root_pos = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 0:3]
        self.root_quat = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 3:7]
        self.root_linvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 7:10]
        self.root_angvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 10:13]
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.body_quat = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self.body_angvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_force_view = self.dof_force.view(self.num_envs, self.num_dofs, 1)[..., 0]
        self.contact_force = self.contact_force.view(self.num_envs, self.num_bodies, 3)[..., 0:3]

        self.arm_dof_pos = self.dof_pos[:, 0:7]
        self.arm_dof_vel = self.dof_vel[:, 0:7]
        self.arm_mass_matrix = self.mass_matrix[:, 0:7, 0:7]  

        self.robot_base_pos = self.body_pos[:, self.robot_base_body_id_env, 0:3]
        self.robot_base_quat = self.body_quat[:, self.robot_base_body_id_env, 0:4]

        self.hand_pos = self.body_pos[:, self.hand_body_id_env, 0:3]
        self.hand_quat = self.body_quat[:, self.hand_body_id_env, 0:4]
        self.hand_linvel = self.body_linvel[:, self.hand_body_id_env, 0:3]
        self.hand_angvel = self.body_angvel[:, self.hand_body_id_env, 0:3]
        self.hand_jacobian = self.jacobian[:, self.hand_body_id_env - self.robot_base_body_id_env - 1, 0:6,
                             0:7]  # minus 1 because base is fixed

        self.left_finger_pos = self.body_pos[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_quat = self.body_quat[:, self.left_finger_body_id_env, 0:4]
        self.left_finger_linvel = self.body_linvel[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_angvel = self.body_angvel[:, self.left_finger_body_id_env, 0:3]
        self.left_finger_jacobian = self.jacobian[:, self.left_finger_body_id_env - self.robot_base_body_id_env - 1,
                                    0:6,
                                    0:7]  # minus 1 because base is fixed

        self.right_finger_pos = self.body_pos[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_quat = self.body_quat[:, self.right_finger_body_id_env, 0:4]
        self.right_finger_linvel = self.body_linvel[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_angvel = self.body_angvel[:, self.right_finger_body_id_env, 0:3]
        self.right_finger_jacobian = self.jacobian[:, self.right_finger_body_id_env - self.robot_base_body_id_env - 1,
                                     0:6,
                                     0:7]  # minus 1 because base is fixed

        self.middle_finger_pos = self.body_pos[:, self.middle_finger_body_id_env, 0:3]
        self.middle_finger_quat = self.body_quat[:, self.middle_finger_body_id_env, 0:4]
        self.middle_finger_linvel = self.body_linvel[:, self.middle_finger_body_id_env, 0:3]
        self.middle_finger_angvel = self.body_angvel[:, self.middle_finger_body_id_env, 0:3]
        self.middle_finger_jacobian = self.jacobian[:, self.middle_finger_body_id_env - self.robot_base_body_id_env - 1,
                                      0:6,
                                      0:7]  # minus 1 because base is fixed

        self.left_finger_force = self.contact_force[:, self.left_finger_body_id_env, 0:3]
        self.right_finger_force = self.contact_force[:, self.right_finger_body_id_env, 0:3]
        self.middle_finger_force = self.contact_force[:, self.middle_finger_body_id_env, 0:3]

        self.gripper_dof_pos = self.dof_pos[:, 7:]
        self.dof_dict = {index: value for index, value in enumerate(self.kuka_joints_names)}

        self.fingertip_centered_pos = self.body_pos[:, self.fingertip_centered_body_id_env, 0:3]
        self.fingertip_centered_quat = self.body_quat[:, self.fingertip_centered_body_id_env, 0:4]
        self.fingertip_centered_linvel = self.body_linvel[:, self.fingertip_centered_body_id_env, 0:3]
        self.fingertip_centered_angvel = self.body_angvel[:, self.fingertip_centered_body_id_env, 0:3]
        self.fingertip_centered_jacobian = self.jacobian[:,
                                           self.fingertip_centered_body_id_env - self.robot_base_body_id_env - 1, 0:6,
                                           0:7]  # minus 1 because base is fixed

        self.fingertip_midpoint_pos = self.fingertip_centered_pos.detach().clone()  # initial value
        self.fingertip_midpoint_quat = self.fingertip_centered_quat  # always equal
        self.fingertip_midpoint_linvel = self.fingertip_centered_linvel.detach().clone()  # initial value

        # From sum of angular velocities
        # (https://physics.stackexchange.com/questions/547698/understanding-addition-of-angular-velocity),
        # angular velocity of midpoint w.r.t. world is equal to sum of
        # angular velocity of midpoint w.r.t. hand and angular velocity of hand w.r.t. world.
        # Midpoint is in sliding contact (i.e., linear relative motion) with hand;
        # angular velocity of midpoint w.r.t. hand is zero.
        # Thus, angular velocity of midpoint w.r.t. world is equal to angular velocity of hand w.r.t. world.

        self.fingertip_midpoint_angvel = self.fingertip_centered_angvel  # always equal
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian + self.middle_finger_jacobian) * 1 / 3  # approximation

        self.dof_torque = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.fingertip_contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.ctrl_target_dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.ctrl_target_gripper_dof_pos = torch.zeros((self.num_envs, self.gripper_dof_pos.shape[-1]), device=self.device)
        self.ctrl_target_fingertip_contact_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.ctrl_target_fingertip_centered_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_centered_quat = torch.zeros((self.num_envs, 4), device=self.device)

        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.gripper_normal_quat = (torch.tensor([-1 / 2 ** 0.5, -1 / 2 ** 0.5, 0.0, 0.0],
                                                 device=self.device).unsqueeze(0).repeat(self.num_envs, 1))

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.gripper_dof_pos_close = torch.zeros_like(self.gripper_dof_pos)
        self.gripper_dof_pos_open = torch.zeros_like(self.gripper_dof_pos)

    def refresh_base_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # Privileged
        self.finger_midpoint_pos = (self.left_finger_pos + self.right_finger_pos + self.middle_finger_pos) * (1 / 3)
        self.fingertip_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.gripper_normal_quat,
                                                                 offset=0.2,
                                                                 device=self.device)

        self.fingertip_midpoint_linvel = self.fingertip_centered_linvel + torch.cross(self.fingertip_centered_angvel,
                                                                                      (self.fingertip_midpoint_pos - self.fingertip_centered_pos),
                                                                                      dim=1)
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian + self.middle_finger_jacobian) * (1 / 3)  # approximation

    def parse_controller_spec(self):
        """Parse controller specification into lower-level controller configuration."""

        cfg_ctrl_keys = {'num_envs',
                         'jacobian_type',
                         'gripper_prop_gains',
                         'gripper_deriv_gains',
                         'motor_ctrl_mode',
                         'gain_space',
                         'ik_method',
                         'joint_prop_gains',
                         'joint_deriv_gains',
                         'do_motion_ctrl',
                         'task_prop_gains',
                         'task_deriv_gains',
                         'do_inertial_comp',
                         'motion_ctrl_axes',
                         'do_force_ctrl',
                         'force_ctrl_method',
                         'wrench_prop_gains',
                         'force_ctrl_axes'}
        self.cfg_ctrl = {cfg_ctrl_key: None for cfg_ctrl_key in cfg_ctrl_keys}

        self.cfg_ctrl['num_envs'] = self.num_envs
        self.cfg_ctrl['jacobian_type'] = self.cfg_task.ctrl.all.jacobian_type
        self.cfg_ctrl['gripper_prop_gains'] = torch.tensor(self.cfg_task.ctrl.all.gripper_prop_gains,
                                                           device=self.device).repeat((self.num_envs, 1))
        self.cfg_ctrl['gripper_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.all.gripper_deriv_gains,
                                                            device=self.device).repeat((self.num_envs, 1))

        ctrl_type = self.cfg_task.ctrl.ctrl_type
        if ctrl_type == 'gym_default':
            self.cfg_ctrl['motor_ctrl_mode'] = 'gym'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.gym_default.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.gym_default.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'joint_space_ik':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_ik.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_ik.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_ik.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
        elif ctrl_type == 'joint_space_id':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_id.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_id.joint_prop_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.joint_space_id.joint_deriv_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
        elif ctrl_type == 'task_space_impedance':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_deriv_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.motion_ctrl_axes,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False
        elif ctrl_type == 'operational_space_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.operational_space_motion.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_deriv_gains, device=self.device).repeat(
                (self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.motion_ctrl_axes, device=self.device).repeat(
                (self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False
        elif ctrl_type == 'open_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'open'
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.open_loop_force.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'closed_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(self.cfg_task.ctrl.closed_loop_force.wrench_prop_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.closed_loop_force.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))
        elif ctrl_type == 'hybrid_force_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.task_deriv_gains,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.motion_ctrl_axes,
                                                             device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.wrench_prop_gains,
                                                              device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(self.cfg_task.ctrl.hybrid_force_motion.force_ctrl_axes,
                                                            device=self.device).repeat((self.num_envs, 1))

        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            prop_gains = torch.cat((self.cfg_ctrl['joint_prop_gains'],
                                    self.cfg_ctrl['gripper_prop_gains']), dim=-1).to('cpu')
            deriv_gains = torch.cat((self.cfg_ctrl['joint_deriv_gains'],
                                     self.cfg_ctrl['gripper_deriv_gains']), dim=-1).to('cpu')
            # No tensor API for getting/setting actor DOF props; thus, loop required
            for env_ptr, kuka_handle, prop_gain, deriv_gain in zip(self.env_ptrs, self.kuka_handles, prop_gains,
                                                                   deriv_gains):
                kuka_dof_props = self.gym.get_actor_dof_properties(env_ptr, kuka_handle)
                kuka_dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
                kuka_dof_props['stiffness'] = prop_gain
                kuka_dof_props['damping'] = deriv_gain
                self.gym.set_actor_dof_properties(env_ptr, kuka_handle, kuka_dof_props)
        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            # No tensor API for getting/setting actor DOF props; thus, loop required
            for env_ptr, kuka_handle in zip(self.env_ptrs, self.kuka_handles):
                kuka_dof_props = self.gym.get_actor_dof_properties(env_ptr, kuka_handle)
                kuka_dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
                kuka_dof_props['stiffness'][:] = 0.0  # zero passive stiffness
                kuka_dof_props['damping'][:] = 0.0  # zero passive damping
                self.gym.set_actor_dof_properties(env_ptr, kuka_handle, kuka_dof_props)

    def generate_ctrl_signals(self):
        """Get Jacobian. Set kuka DOF position targets or DOF torques."""

        # Get desired Jacobian
        if self.cfg_ctrl['jacobian_type'] == 'geometric':
            self.fingertip_centered_jacobian_tf = self.fingertip_centered_jacobian

        elif self.cfg_ctrl['jacobian_type'] == 'analytic':
            self.fingertip_centered_jacobian_tf = fc.get_analytic_jacobian(
                fingertip_quat=self.fingertip_centered_quat,
                fingertip_jacobian=self.fingertip_centered_jacobian,
                num_envs=self.num_envs,
                device=self.device)

        # Set PD joint pos target or joint torque
        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            self._set_dof_pos_target()
        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            self._set_dof_torque()

    def _set_dof_pos_target(self):
        """Set kuka DOF position target to move fingertips towards target pose."""

        self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl=self.cfg_ctrl,
            arm_dof_pos=self.arm_dof_pos,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            jacobian=self.fingertip_centered_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            device=self.device)

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ctrl_target_dof_pos),
                                                        gymtorch.unwrap_tensor(self.kuka_actor_ids_sim),
                                                        len(self.kuka_actor_ids_sim))

    def _set_dof_torque(self):
        """Set kuka DOF torque to move fingertips towards target pose."""

        self.dof_torque = fc.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            fingertip_midpoint_linvel=self.fingertip_centered_linvel,
            fingertip_midpoint_angvel=self.fingertip_centered_angvel,
            left_finger_force=self.left_finger_force,
            right_finger_force=self.right_finger_force,
            jacobian=self.fingertip_centered_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_fingertip_contact_wrench=self.ctrl_target_fingertip_contact_wrench,
            device=self.device)

        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_torque),
                                                        gymtorch.unwrap_tensor(self.kuka_actor_ids_sim),
                                                        len(self.kuka_actor_ids_sim))

    def print_sdf_warning(self):
        """Generate SDF warning message."""

        logger.warn('Please be patient: SDFs may be generating, which may take a few minutes. Terminating prematurely may result in a corrupted SDF cache.')

    def print_sdf_finish(self):
        """Generate SDF warning message."""

        logger.warn(
            'Finished generating SDFS.')
    def simulate_and_refresh(self):
        """
            Simulate one step, refresh tensors, and render results.
        """

        self.gym.simulate(self.sim)
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.render()
    
    def enable_gravity(self):
        """Enable gravity."""
        xyz_gravity = self.cfg_base.sim.gravity
        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity.z = xyz_gravity[2]
        sim_params.gravity.x = xyz_gravity[0]
        sim_params.gravity.y = xyz_gravity[1]
        self.gym.set_sim_params(self.sim, sim_params)

    def disable_gravity(self):
        """Disable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity.x = 0.0
        sim_params.gravity.y = 0.0
        sim_params.gravity.z = 0.0
        self.gym.set_sim_params(self.sim, sim_params)

    def export_scene(self, label):
        """Export scene to USD."""

        usd_export_options = gymapi.UsdExportOptions()
        usd_export_options.export_physics = False

        usd_exporter = self.gym.create_usd_exporter(usd_export_options)
        self.gym.export_usd_sim(usd_exporter, self.sim, label)
        sys.exit()


    def open_gripper(self, sim_steps):
        """Open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self.move_gripper_to_target_pose(gripper_dof_pos=self.gripper_dof_pos_open, sim_steps=sim_steps)

    def close_gripper(self, sim_steps):
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        sub_steps = 5
        for _ in range(int(sim_steps/sub_steps)):
            diff = self.gripper_dof_pos_close - self.gripper_dof_pos
            self.move_gripper_to_target_pose(gripper_dof_pos=self.gripper_dof_pos + diff * 0.6, sim_steps=sub_steps)

    def clamp_pose_error(self, pos_error, axis_angle_error, dx, dr):
        """
        Post-processes the pose error by clamping position and rotation displacements.

        Args:
            pose_error (torch.Tensor): N x 6 tensor where the first 3 columns are position error,
                                    and the last 3 columns are rotation error in axis-angle.
            dx (float): Maximum position displacement.
            dr (float): Maximum rotation displacement in radians.

        Returns:
            torch.Tensor: Clamped pose error with position and rotation displacements within the thresholds.
        """

        # Clamp the position error to be within the threshold dx
        position_error_clamped = torch.clamp(pos_error.clone(), min=-dx, max=dx)

        # Compute the magnitude of the rotation error (axis-angle norm)
        rotation_magnitude = torch.norm(axis_angle_error.clone(), dim=1, keepdim=True)  # N x 1

        # Scale the rotation error if its magnitude exceeds dr
        scale_factor = torch.minimum(rotation_magnitude / dr, torch.ones_like(rotation_magnitude))
        rotation_error_clamped = axis_angle_error.clone() * scale_factor

        return position_error_clamped, rotation_error_clamped

    def move_gripper_to_target_pose(self, gripper_dof_pos, sim_steps):
        """Move gripper to control target pose."""

        ctrl_target_fingertip_centered_pos = self.ctrl_target_fingertip_centered_pos.clone()
        ctrl_target_fingertip_centered_quat = self.ctrl_target_fingertip_centered_quat.clone()
        for _ in range(sim_steps):
            # NOTE: midpoint is calculated based on the midpoint between the actual gripper finger pos, 
            # and centered is calculated with the assumption that the gripper fingers are perfectly mirrored.
            # Here we **intentionally** use *_centered_* pos and quat instead of *_midpoint_*,
            # since the fingertips are exactly mirrored in the real world.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_centered_pos,
                fingertip_midpoint_quat=self.fingertip_centered_quat,
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_centered_pos, 
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_centered_quat, 
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            # pos_error, axis_angle_error = self.clamp_pose_error(pos_error, axis_angle_error, 0.02, 0.08726646)
            # pos_error, axis_angle_error = self.clamp_pose_error(pos_error, axis_angle_error, 0.02, 0.17453292)

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=gripper_dof_pos,
                do_scale=False,
            )

            ## only for visualization
            fingertip_centered_pose = torch.cat((self.fingertip_centered_pos, self.fingertip_centered_quat), dim=-1)
            ctrl_target_fingertip_centered_pose = torch.cat(
                (self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat), dim=-1
            )
            self.visualize_poses(torch.cat((fingertip_centered_pose[:, None, :], ctrl_target_fingertip_centered_pose[:, None, :]), dim=1))

            # Simulate one step
            self.simulate_and_refresh()

        # Stabilize Kuka
        self.dof_vel[:, :] = 0.0
        self.dof_torque[:, :] = 0.0
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()

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

    def pose_world_to_robot_base(self, pos, quat):
        """Convert pose from world frame to robot base frame."""

        robot_base_transform_inv = torch_utils.tf_inverse(
            self.robot_base_quat, self.robot_base_pos
        )
        quat_in_robot_base, pos_in_robot_base = torch_utils.tf_combine(
            robot_base_transform_inv[0], robot_base_transform_inv[1], quat, pos
        )

        return pos_in_robot_base, quat_in_robot_base

    def pose_robot_base_to_world(self, pos, quat):
        """Convert pose from robot base frame to world frame."""

        quat_in_world, pos_in_world = torch_utils.tf_combine(
            self.robot_base_quat, self.robot_base_pos, quat, pos
        )

        return pos_in_world, quat_in_world