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

"""Wrench, nut and bolt asset loading and scene construction."""

import hydra
import math
import numpy as np
import os
import torch

from isaacgym import gymapi
from rhythmic_insertion.sim.tasks.industreal_wrench.schema.factory_schema_class_env import FactoryABCEnv
from rhythmic_insertion.sim.tasks.industreal_wrench.schema.factory_schema_config_env import FactorySchemaConfigEnv
from rhythmic_insertion.sim.tasks.industreal_wrench.kuka_base import IndustRealKukaBase

from rhythmic_insertion.sim.utils import torch_jit_utils

class IndustRealEnvWrench(IndustRealKukaBase, FactoryABCEnv):
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
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_env_yaml_params()

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()
        self._load_predefined_gripper_states()

    def _load_predefined_gripper_states(self):
        ## wrench1
        positions_indices1 = [
            ('palm_finger_1_joint', 0, 0),
            ('palm_finger_2_joint', 0, 0),
            ('finger_1_joint_1', 0, 1.2),
            ('finger_2_joint_1', 0, 1.2),
            ('finger_middle_joint_1', 0, 1.2),
            ('finger_1_joint_2', 0, 1.2),
            ('finger_2_joint_2', 0, 1.2),
            ('finger_middle_joint_2', 0, 1.2),
            ('finger_1_joint_3', 0, 0),
            ('finger_2_joint_3', 0, 0),
            ('finger_middle_joint_3', 0, 0),
        ]
        ## wrench2
        positions_indices2 = [
            ('palm_finger_1_joint', 0, 0),
            ('palm_finger_2_joint', 0, 0),
            ('finger_1_joint_1', 0, 1.22),
            ('finger_2_joint_1', 0, 1.22),
            ('finger_middle_joint_1', 0, 1.2),
            ('finger_1_joint_2', 0, 1.2),
            ('finger_2_joint_2', 0, 1.2),
            ('finger_middle_joint_2', 0, 1.2),
            ('finger_1_joint_3', 0, 0),
            ('finger_2_joint_3', 0, 0),
            ('finger_middle_joint_3', 0, 0),
        ]
        ## wrench3
        positions_indices3 = [
            ('palm_finger_1_joint', 0, 0),
            ('palm_finger_2_joint', 0, 0),
            ('finger_1_joint_1', 0, 1.15),
            ('finger_2_joint_1', 0, 1.15),
            ('finger_middle_joint_1', 0, 1.15),
            ('finger_1_joint_2', 0, 1.2),
            ('finger_2_joint_2', 0, 1.2),
            ('finger_middle_joint_2', 0, 1.2),
            ('finger_1_joint_3', 0, 0),
            ('finger_2_joint_3', 0, 0),
            ('finger_middle_joint_3', 0, 0),
        ]
        ## wrench4
        positions_indices4 = [
            ('palm_finger_1_joint', 0, 0),
            ('palm_finger_2_joint', 0, 0),
            ('finger_1_joint_1', 0, 0.85),
            ('finger_2_joint_1', 0, 0.85),
            ('finger_middle_joint_1', 0, 0.85),
            ('finger_1_joint_2', 0, 1.7),
            ('finger_2_joint_2', 0, 1.7),
            ('finger_middle_joint_2', 0, 1.7),
            ('finger_1_joint_3', 0, 0),
            ('finger_2_joint_3', 0, 0),
            ('finger_middle_joint_3', 0, 0),
        ]
        ## wrench5
        positions_indices5 = [
            ('palm_finger_1_joint', 0, 0),
            ('palm_finger_2_joint', 0, 0),
            ('finger_1_joint_1', 0, 0.85),
            ('finger_2_joint_1', 0, 0.85),
            ('finger_middle_joint_1', 0, 0.85),
            ('finger_1_joint_2', 0, 1.7),
            ('finger_2_joint_2', 0, 1.7),
            ('finger_middle_joint_2', 0, 1.7),
            ('finger_1_joint_3', 0, 0),
            ('finger_2_joint_3', 0, 0),
            ('finger_middle_joint_3', 0, 0),
        ]
        wrench_gripper_positions = {
            'nut1_bolt1_wrench1': positions_indices1,
            'nut2_bolt2_wrench2': positions_indices2,
            'nut3_bolt3_wrench3': positions_indices3,
            'nut4_bolt4_wrench4': positions_indices4,
            'nut5_bolt5_wrench5': positions_indices5,
        }

        for env_i, j in enumerate(self.asset_indices):
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            for dof, open_value, close_value in wrench_gripper_positions[subassembly]:
                index = list(self.dof_dict.values()).index(dof) - 7
                self.gripper_dof_pos_close[env_i, index] = close_value
                self.gripper_dof_pos_open[env_i, index] = open_value

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = "task/IndustRealKukaEnvWrench.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting
        if "desired_subassemblies" in self.cfg["env"]:
            self.cfg_env.env.desired_subassemblies = self.cfg["env"]["desired_subassemblies"]

        asset_info_path = "../../assets/screw/yaml/screw_asset_info_nut_bolt_wrench.low.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion[""][""][""][""][""][""][
            "assets"
        ]["screw"][
            "yaml"
        ]  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(
            -self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0
        )
        upper = gymapi.Vec3(
            self.cfg_base.env.env_spacing,
            self.cfg_base.env.env_spacing,
            self.cfg_base.env.env_spacing,
        )
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        kuka_asset, table_asset = self.import_kuka_assets()
        plug_assets, socket_assets, bolt_assets = self._import_env_assets()
        self._create_actors(
            lower,
            upper,
            num_per_row,
            kuka_asset,
            plug_assets,
            socket_assets,
            bolt_assets,
            table_asset,
        )
        self.print_sdf_finish()
        self.camera_handle = None
        if self.cfg["env"].get("enableCameraSensors", False):
            props = gymapi.CameraProperties()
            props.width = props.height = int(self.cfg["env"].get("camera_resolution", 256))
            props.horizontal_fov = 60.0
            props.enable_tensors = False
            self.camera_handle = self.gym.create_camera_sensor(self.env_ptrs[0], props)
            if self.camera_handle < 0:
                raise RuntimeError("Could not create the insertion camera; check the graphics device.")
            self.gym.set_camera_location(
                self.camera_handle, self.env_ptrs[0],
                gymapi.Vec3(0.35, -0.4, 1.0), gymapi.Vec3(0.0, 0.0, 0.55),
            )

    def _import_env_assets(self):
        """Set plug (wrench), socket (nut), and bolt asset options. Import assets."""

        self.plug_files, self.socket_files, self.bolt_files = [], [], []

        urdf_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "assets", "screw", "urdf"
        )

        ## Wrench
        plug_options = gymapi.AssetOptions()
        plug_options.flip_visual_attachments = False
        plug_options.fix_base_link = False
        plug_options.thickness = 0.0  
        plug_options.armature = 0.0  
        plug_options.use_physx_armature = True
        plug_options.linear_damping = 0.5  
        plug_options.max_linear_velocity = 1000.0  
        plug_options.angular_damping = 0.5  
        plug_options.max_angular_velocity = 64.0  
        plug_options.disable_gravity = True # False
        plug_options.enable_gyroscopic_forces = True
        plug_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        plug_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            plug_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        ## Nut
        socket_options = gymapi.AssetOptions()
        socket_options.flip_visual_attachments = False
        socket_options.fix_base_link = self.cfg_env.env.is_socket_fixed # True
        socket_options.thickness = 0.0  
        socket_options.armature = 0.0  
        socket_options.use_physx_armature = True
        socket_options.linear_damping = 0.0  
        socket_options.max_linear_velocity = 1000.0  
        socket_options.angular_damping = 0.0  
        socket_options.max_angular_velocity = 64  
        socket_options.disable_gravity = False
        socket_options.enable_gyroscopic_forces = True
        socket_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        socket_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            socket_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        ## Bolt
        bolt_options = gymapi.AssetOptions()
        bolt_options.flip_visual_attachments = False
        bolt_options.fix_base_link = True
        bolt_options.thickness = 0.0  
        bolt_options.armature = 0.0  
        bolt_options.use_physx_armature = True
        bolt_options.linear_damping = 0.0  
        bolt_options.max_linear_velocity = 1000.0  
        bolt_options.angular_damping = 0.0  
        bolt_options.max_angular_velocity = 64.0  
        bolt_options.disable_gravity = False
        bolt_options.enable_gyroscopic_forces = True
        bolt_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        bolt_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            bolt_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        plug_assets = []
        socket_assets = []
        bolt_assets = []
        for subassembly in self.cfg_env.env.desired_subassemblies:
            components = list(self.asset_info_insertion[subassembly])
            # Wrench
            plug_file = (
                self.asset_info_insertion[subassembly][components[2]]["urdf_path"]
                + ".urdf"
            )
            # Nut
            socket_file = (
                self.asset_info_insertion[subassembly][components[0]]["urdf_path"]
                + ".urdf"
            )
            # Bolt
            bolt_file = (
                self.asset_info_insertion[subassembly][components[1]]["urdf_path"]
                + ".urdf"
            )
            plug_options.density = self.asset_info_insertion[subassembly][
                components[2]
            ]["density"]
            socket_options.density = self.asset_info_insertion[subassembly][
                components[0]
            ]["density"]
            bolt_options.density = self.asset_info_insertion[subassembly][
                components[1]
            ]["density"]
            plug_asset = self.gym.load_asset(
                self.sim, urdf_root, plug_file, plug_options
            )
            socket_asset = self.gym.load_asset(
                self.sim, urdf_root, socket_file, socket_options
            )
            bolt_asset = self.gym.load_asset(
                self.sim, urdf_root, bolt_file, bolt_options
            )
            plug_assets.append(plug_asset)
            socket_assets.append(socket_asset)
            bolt_assets.append(bolt_asset)

            # Save URDF file paths (for loading appropriate meshes during SAPU and SDF-Based Reward calculations)
            self.plug_files.append(os.path.join(urdf_root, plug_file))
            self.socket_files.append(os.path.join(urdf_root, socket_file))
            self.bolt_files.append(os.path.join(urdf_root, bolt_file))

        return plug_assets, socket_assets, bolt_assets

    def _create_actors(
        self,
        lower,
        upper,
        num_per_row,
        kuka_asset,
        plug_assets,
        socket_assets,
        bolt_assets,
        table_asset,
    ):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""
        # NOTE: Closely adapted from FactoryEnvInsertion; however, plug grasp offsets, plug widths, socket heights,
        # and asset indices are now stored for possible use during policy learning."""

        kuka_pose = gymapi.Transform()
        kuka_pose.p.x = self.asset_info_kuka_table.table_depth * 0.5 +  self.cfg_base.env.kuka_depth
        kuka_pose.p.y = 0.0
        kuka_pose.p.z = self.cfg_base.env.table_height + self.cfg_base.env.kuka_height
        kuka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.kuka_handles = []
        self.plug_handles = []
        self.socket_handles = []
        self.bolt_handles = []
        self.table_handles = []
        self.shape_ids = []
        self.kuka_actor_ids_sim = []  # within-sim indices
        self.plug_actor_ids_sim = []  # within-sim indices
        self.socket_actor_ids_sim = []  # within-sim indices
        self.bolt_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        self.socket_heights = []
        self.plug_thicknesses = []
        self.asset_indices = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            ### 1. create actors ###
            ## 1.1 kuka robot
            if self.cfg_base.sim.disable_kuka_collisions:
                kuka_handle = self.gym.create_actor(env_ptr, kuka_asset, kuka_pose, 'kuka', i + self.num_envs, 1, 0)
            else:
                kuka_handle = self.gym.create_actor(env_ptr, kuka_asset, kuka_pose, 'kuka', i, 1, 0)
            self.kuka_actor_ids_sim.append(actor_count)
            actor_count += 1

            ## 1.2 nut-bolt-wrench pair
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_insertion[subassembly])

            # wrench
            plug_pose = gymapi.Transform()
            plug_pose.p.x = 0.0
            plug_pose.p.y = 0.0
            plug_pose.p.z = self.cfg_base.env.table_height + self.asset_info_insertion[subassembly][components[2]]['thickness']
            plug_pose.r = gymapi.Quat(0.0, 0.0, -0.707, 0.707)
            plug_handle = self.gym.create_actor(
                env_ptr, plug_assets[j], plug_pose, "plug", i, 0, 0
            )
            self.plug_actor_ids_sim.append(actor_count)
            actor_count += 1

            # nut
            socket_pose = gymapi.Transform()
            socket_pose.p.x = 0.0
            socket_pose.p.y = 0.0
            socket_pose.p.z = self.cfg_base.env.table_height + self.asset_info_insertion[subassembly][components[1]]['shank_length'] + 0.1
            socket_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            socket_handle = self.gym.create_actor(
                env_ptr, socket_assets[j], socket_pose, "socket", i, 0, 0
            )
            self.socket_actor_ids_sim.append(actor_count)
            actor_count += 1

            # bolt
            bolt_pose = gymapi.Transform()
            bolt_pose.p.x = 0.0
            bolt_pose.p.y = 0.0
            bolt_pose.p.z = self.cfg_base.env.table_height + 0.1
            bolt_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            bolt_handle = self.gym.create_actor(env_ptr, bolt_assets[j], bolt_pose, 'bolt', i, 0, 0)
            self.bolt_actor_ids_sim.append(actor_count)
            actor_count += 1

            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 0, 0
            )
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            ### 2. set shape properties ###
            r"""
            iiwa7_link_0             :   0
            iiwa7_link_1             :   1
            iiwa7_link_2             :   2
            iiwa7_link_3             :   3
            iiwa7_link_4             :   4
            iiwa7_link_5             :   5
            iiwa7_link_6             :   6
            iiwa7_link_7             :   7
            dummy_ft_link            :   8
            finger_1_link_0          :  11
            finger_1_link_1          :  12
            finger_1_link_2          :  13
            finger_1_link_3          :  14
            finger_2_link_0          :  15
            finger_2_link_1          :  16
            finger_2_link_2          :  17
            finger_2_link_3          :  18
            finger_middle_link_0     :  19
            finger_middle_link_1     :  20
            finger_middle_link_2     :  21
            finger_middle_link_3     :  22
            palm                     :   9
            kuka_fingertip_centered  :  10 (no shape properties)
            """

            ## 2.1 robot eef
            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'iiwa7_link_7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'palm', gymapi.DOMAIN_ACTOR) # ROBOTIQ
            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_1_link_3', gymapi.DOMAIN_ACTOR) # ROBOTIQ
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_2_link_3', gymapi.DOMAIN_ACTOR) # ROBOTIQ
            middle_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_middle_link_3', gymapi.DOMAIN_ACTOR) # ROBOTIQ
            # useful for measuring the friction parameters (privileged information)
            self.left_finger_id = left_finger_id - 1
            self.right_finger_id = right_finger_id - 1
            self.middle_finger_id = middle_finger_id - 1
            self.shape_ids = [link7_id, hand_id, left_finger_id - 1, right_finger_id - 1, middle_finger_id - 1]

            kuka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, kuka_handle)
            for shape_id in self.shape_ids:
                kuka_shape_props[shape_id].friction = self.cfg_base.env.kuka_friction
                kuka_shape_props[shape_id].rolling_friction = 0.0  
                kuka_shape_props[shape_id].torsion_friction = 0.0  
                kuka_shape_props[shape_id].restitution = 0.0  
                kuka_shape_props[shape_id].compliance = 0.0 # 0.0  # default = 0.0
                kuka_shape_props[shape_id].thickness = 0.0  
            self.gym.set_actor_rigid_shape_properties(env_ptr, kuka_handle, kuka_shape_props)

            ## 2.2 plug(wrench), socket(nut), bolt
            plug_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, plug_handle
            )
            plug_shape_props[0].friction = self.asset_info_insertion[subassembly][
                components[2]
            ]["friction"]
            plug_shape_props[0].rolling_friction = 0.0  
            plug_shape_props[0].torsion_friction = 0.0  
            plug_shape_props[0].restitution = 0.0  
            plug_shape_props[0].compliance = 0.2  
            plug_shape_props[0].thickness = 0.0  
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, plug_handle, plug_shape_props
            )

            socket_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, socket_handle
            )
            socket_shape_props[0].friction = self.asset_info_insertion[subassembly][
                components[0]
            ]["friction"]
            socket_shape_props[0].rolling_friction = 0.0  
            socket_shape_props[0].torsion_friction = 0.0  
            socket_shape_props[0].restitution = 0.0  
            socket_shape_props[0].compliance = 0.0  
            socket_shape_props[0].thickness = 0.0  
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, socket_handle, socket_shape_props
            )

            bolt_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, bolt_handle)
            bolt_shape_props[0].friction = self.asset_info_insertion[subassembly][
                components[1]
            ]["friction"]
            bolt_shape_props[0].rolling_friction = 0.0  
            bolt_shape_props[0].torsion_friction = 0.0  
            bolt_shape_props[0].restitution = 0.0  
            bolt_shape_props[0].compliance = 0.0  
            bolt_shape_props[0].thickness = 0.0  
            self.gym.set_actor_rigid_shape_properties(env_ptr, bolt_handle, bolt_shape_props)

            ## 2.3 table
            table_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, table_handle
            )
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  
            table_shape_props[0].torsion_friction = 0.0  
            table_shape_props[0].restitution = 0.0  
            table_shape_props[0].compliance = 0.0  
            table_shape_props[0].thickness = 0.0  
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, table_handle, table_shape_props
            )

            ### 3. set DOF properties ###
            self.kuka_num_dofs = self.gym.get_actor_dof_count(env_ptr, kuka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, kuka_handle)

            socket_height = self.asset_info_insertion[subassembly][components[0]][
                "height"
            ]
            plug_thickness = self.asset_info_insertion[subassembly][components[2]][
                "thickness"
            ]

            self.env_ptrs.append(env_ptr)
            self.kuka_handles.append(kuka_handle)
            self.plug_handles.append(plug_handle)
            self.socket_handles.append(socket_handle)
            self.bolt_handles.append(bolt_handle)
            self.table_handles.append(table_handle)

            self.socket_heights.append(socket_height)
            self.plug_thicknesses.append(plug_thickness)
            self.asset_indices.append(j)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        self.kuka_actor_ids_sim = torch.tensor(
            self.kuka_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.plug_actor_ids_sim = torch.tensor(
            self.plug_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.socket_actor_ids_sim = torch.tensor(
            self.socket_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.bolt_actor_ids_sim = torch.tensor(self.bolt_actor_ids_sim, dtype=torch.int32, device=self.device)

        self.plug_actor_id_env = self.gym.find_actor_index(
            env_ptr, "plug", gymapi.DOMAIN_ENV
        )
        self.socket_actor_id_env = self.gym.find_actor_index(
            env_ptr, "socket", gymapi.DOMAIN_ENV
        )
        self.bolt_actor_id_env = self.gym.find_actor_index(env_ptr, 'bolt', gymapi.DOMAIN_ENV)

        ## BODY ID (Name defined in URDF)
        
        self.robot_base_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, "iiwa7_link_0",
                                                                           gymapi.DOMAIN_ENV)
        self.plug_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, plug_handle, "wrench", gymapi.DOMAIN_ENV
        )
        self.socket_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, socket_handle, "nut", gymapi.DOMAIN_ENV
        )
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'palm',
                                                                     gymapi.DOMAIN_ENV)
        self.wrist_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'dummy_ft_link',
                                                                      gymapi.DOMAIN_ENV) # ROBOTIQ
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_1_link_3',
                                                                            gymapi.DOMAIN_ENV) # ROBOTIQ
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_2_link_3',
                                                                             gymapi.DOMAIN_ENV) # ROBOTIQ
        self.middle_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle, 'finger_middle_link_3',
                                                                              gymapi.DOMAIN_ENV) # ROBOTIQ
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, kuka_handle,
                                                                                   'kuka_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)
        self.wrench_head_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, plug_handle, 'wrench_head',
                                                                       gymapi.DOMAIN_ENV)

        self.kuka_joints_names = self.gym.get_asset_dof_names(kuka_asset)

        self.socket_heights = torch.tensor(self.socket_heights, device=self.device)
        self.plug_thicknesses = torch.tensor(self.plug_thicknesses, device=self.device)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        # wrench
        self.plug_pos = self.root_pos[:, self.plug_actor_id_env, 0:3]
        self.plug_quat = self.root_quat[:, self.plug_actor_id_env, 0:4]
        self.plug_linvel = self.root_linvel[:, self.plug_actor_id_env, 0:3]
        self.plug_angvel = self.root_angvel[:, self.plug_actor_id_env, 0:3]

        # nut
        self.socket_pos = self.root_pos[:, self.socket_actor_id_env, 0:3]
        self.socket_quat = self.root_quat[:, self.socket_actor_id_env, 0:4]
        self.socket_linvel = self.root_linvel[:, self.socket_actor_id_env, 0:3]
        self.socket_angvel = self.root_angvel[:, self.socket_actor_id_env, 0:3]

        # bolt
        self.bolt_pos = self.root_pos[:, self.bolt_actor_id_env, 0:3]
        self.bolt_quat = self.root_quat[:, self.bolt_actor_id_env, 0:4]
        self.bolt_linvel = self.root_linvel[:, self.bolt_actor_id_env, 0:3]
        self.bolt_angvel = self.root_angvel[:, self.bolt_actor_id_env, 0:3]

        # wrench head
        self.wrench_head_pos = self.body_pos[:, self.wrench_head_body_id_env, 0:3] # （num_envs, 3）
        self.wrench_head_quat = self.body_quat[:, self.wrench_head_body_id_env, 0:4] # （num_envs, 4）
        self.wrench_head_linvel = self.body_linvel[:, self.wrench_head_body_id_env, 0:3] # （num_envs, 3）
        self.wrench_head_angvel = self.body_angvel[:, self.wrench_head_body_id_env, 0:3] # （num_envs, 3）

        wrench_head_quat_inv, wrench_head_pos_inv = torch_jit_utils.tf_inverse(self.wrench_head_quat, self.wrench_head_pos)
        plug_quat_inv, plug_pos_inv = torch_jit_utils.tf_inverse(self.plug_quat, self.plug_pos)
        """wrench to wrench head transformation"""
        self.plug_to_wrench_head_quat, self.plug_to_wrench_head_pos = torch_jit_utils.tf_combine(wrench_head_quat_inv, wrench_head_pos_inv, self.plug_quat, self.plug_pos)

        """gripper fingertip midpoint to wrench head transformation"""
        self.fingertip_midpoint_to_wrench_head_quat, self.fingertip_midpoint_to_wrench_head_pos = torch_jit_utils.tf_combine(wrench_head_quat_inv, wrench_head_pos_inv, self.fingertip_centered_quat, self.fingertip_centered_pos)

        """gripper fingertip midpoint to wrench transformation"""
        self.fingertip_midpoint_to_plug_quat, self.fingertip_midpoint_to_plug_pos = torch_jit_utils.tf_combine(plug_quat_inv, plug_pos_inv, self.fingertip_centered_quat, self.fingertip_centered_pos)

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        wrench_head_quat_inv, wrench_head_pos_inv = torch_jit_utils.tf_inverse(self.wrench_head_quat, self.wrench_head_pos)
        plug_quat_inv, plug_pos_inv = torch_jit_utils.tf_inverse(self.plug_quat, self.plug_pos)        
        """wrench to wrench head transformation"""
        self.plug_to_wrench_head_quat, self.plug_to_wrench_head_pos = torch_jit_utils.tf_combine(wrench_head_quat_inv, wrench_head_pos_inv, self.plug_quat, self.plug_pos)

        """gripper fingertip midpoint to wrench head transformation"""
        self.fingertip_midpoint_to_wrench_head_quat, self.fingertip_midpoint_to_wrench_head_pos = torch_jit_utils.tf_combine(wrench_head_quat_inv, wrench_head_pos_inv, self.fingertip_centered_quat, self.fingertip_centered_pos)

        """gripper fingertip midpoint to wrench transformation"""
        self.fingertip_midpoint_to_plug_quat, self.fingertip_midpoint_to_plug_pos = torch_jit_utils.tf_combine(plug_quat_inv, plug_pos_inv, self.fingertip_centered_quat, self.fingertip_centered_pos)
