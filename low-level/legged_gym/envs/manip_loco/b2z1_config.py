# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from .b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO


class B2Z1RoughCfg(B1Z1RoughCfg):
    class goal_ee(B1Z1RoughCfg.goal_ee):
        class sphere_center(B1Z1RoughCfg.goal_ee.sphere_center):
            x_offset = 0.2
            z_invariant_offset = 0.8

        class ranges(B1Z1RoughCfg.goal_ee.ranges):
            pos_l = [0.45, 0.95]
            pos_y = [-0.75, 0.75]

    class env(B1Z1RoughCfg.env):
        num_gripper_joints = 0

    class init_state(B1Z1RoughCfg.init_state):
        pos = [0.0, 0.0, 0.55]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,

            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,

            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,

            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,

            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        }

    class control(B1Z1RoughCfg.control):
        stiffness = {
            "_joint": 250.0,
            "joint1": 50.0,
            "joint2": 50.0,
            "joint3": 80.0,
            "joint4": 30.0,
            "joint5": 30.0,
            "joint6": 20.0,
        }
        damping = {
            "_joint": 5.0,
            "joint1": 3.0,
            "joint2": 2.0,
            "joint3": 3.0,
            "joint4": 3.0,
            "joint5": 2.5,
            "joint6": 1.0,
        }
        action_scale = [0.25] * 18

    class asset(B1Z1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b2_z1/urdf/b2_z1.urdf'
        base_name = "base_link"
        foot_name = "foot"
        gripper_name = "gripperMover"
        penalize_contacts_on = ["thigh", "base_link", "calf"]
        collapse_fixed_joints = False

    class arm(B1Z1RoughCfg.arm):
        base_offset = [0.0, 0.0, 0.09]

    class rewards(B1Z1RoughCfg.rewards):
        base_height_target = 0.48


class B2Z1RoughCfgPPO(B1Z1RoughCfgPPO):
    class policy(B1Z1RoughCfgPPO.policy):
        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains

    class algorithm(B1Z1RoughCfgPPO.algorithm):
        torque_supervision = B2Z1RoughCfg.control.torque_supervision
        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains

    class runner(B1Z1RoughCfgPPO.runner):
        experiment_name = 'b2z1_v2'
