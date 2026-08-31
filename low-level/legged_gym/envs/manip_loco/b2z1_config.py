# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from .b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO
import numpy as np


class B2Z1RoughCfg(B1Z1RoughCfg):
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

            # Keep the Z1 arm in the same neutral pose used by B1Z1.  The
            # B2 URDF uses joint1..joint6 for the same six Z1 axes.
            "joint1": 0.0,
            "joint2": 1.48,
            "joint3": -0.63,
            "joint4": -0.84,
            "joint5": 0.0,
            "joint6": 1.57,
            "z1_jointGripper": -0.7853981633974483,
        }

    class control(B1Z1RoughCfg.control):
        # B2 gains follow Unitree's Isaac Lab asset; Z1 gains follow its ROS PID.
        # Z1 is actually driven by the B1Z1 IK position drive in ManipLoco.
        stiffness = {
            "hip_joint": 160.0,
            "thigh_joint": 160.0,
            "calf_joint": 160.0,
            "joint1": 300.0,
            "joint2": 300.0,
            "joint3": 300.0,
            "joint4": 300.0,
            "joint5": 300.0,
            "joint6": 300.0,
        }
        damping = {
            "hip_joint": 5.0,
            "thigh_joint": 5.0,
            "calf_joint": 5.0,
            "joint1": 5.0,
            "joint2": 5.0,
            "joint3": 5.0,
            "joint4": 5.0,
            "joint5": 5.0,
            "joint6": 5.0,
        }
        action_scale = [0.4, 0.45, 0.45] * 4 + [0.25] * 6

    class asset(B1Z1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b2_z1/urdf/b2_z1.urdf'
        base_name = "base_link"
        foot_name = "foot"
        gripper_name = "ee_gripper_link"
        penalize_contacts_on = ["thigh", "base_link", "calf"]
        collapse_fixed_joints = False

    class arm(B1Z1RoughCfg.arm):
        base_offset = [0.0, 0.0, 0.09]

    class rewards(B1Z1RoughCfg.rewards):
        base_height_target = 0.55

        class scales(B1Z1RoughCfg.rewards.scales):
            walking_dof = 1.0
            tracking_lin_vel_max = 2.5
            torques = -1.0e-5

class B2Z1RoughCfgPPO(B1Z1RoughCfgPPO):
    class policy(B1Z1RoughCfgPPO.policy):
        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains

    class algorithm(B1Z1RoughCfgPPO.algorithm):
        torque_supervision = B2Z1RoughCfg.control.torque_supervision
        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains

    class runner(B1Z1RoughCfgPPO.runner):
        experiment_name = 'b2z1_v2'


class B2Z1BoundedActionsCfg(B2Z1RoughCfg):
    """B2-Z1 locomotion task with gait observations enabled by default."""

    class env(B2Z1RoughCfg.env):
        observe_gait_commands = True


class B2Z1BoundedActionsCfgPPO(B2Z1RoughCfgPPO):
    class policy(B2Z1RoughCfgPPO.policy):
        output_tanh = True

    class runner(B2Z1RoughCfgPPO.runner):
        experiment_name = 'b2_z1_bounded_actions'

class B2Z1ReachableWorkspaceCfg(B2Z1BoundedActionsCfg):
    """B2-Z1 task with reachable arm targets and balanced locomotion rewards."""

    class goal_ee(B2Z1BoundedActionsCfg.goal_ee):
        class sphere_center(B2Z1BoundedActionsCfg.goal_ee.sphere_center):
            x_offset = 0.0
            z_invariant_offset = 0.8

        class ranges(B2Z1BoundedActionsCfg.goal_ee.ranges):
            pos_l = [0.45, 0.82]
            pos_p = [-1.00, 0.80]
            pos_y = [-1.2, 1.2]

            # 原始 b1z1 版本
            # delta_orn_r = [-0.5, 0.5]
            # delta_orn_p = [-0.5, 0.5]
            # delta_orn_y = [-0.5, 0.5]

            # DQ-NET b1z1 版本 
            # delta_orn_r = [-1.5, 1.5]
            # delta_orn_p = [-1.2, 1.6]
            # delta_orn_y = [-0.8, 0.8]

    class rewards(B2Z1BoundedActionsCfg.rewards):
        class scales(B2Z1BoundedActionsCfg.rewards.scales):
            walking_dof = 0.9
            tracking_lin_vel_max = 2.75
            tracking_ang_vel = 0.5
            # feet_height = 0.0


class B2Z1ReachableWorkspaceCfgPPO(B2Z1BoundedActionsCfgPPO):
    class runner(B2Z1BoundedActionsCfgPPO.runner):
        experiment_name = 'b2_z1_reachable_workspace'
        save_interval = 500


class B2Z1ReachableWorkspaceMotionCfg(B2Z1ReachableWorkspaceCfg):
    """Reachable workspace variant with moderate whole-body locomotion motion.

    Compared to B2Z1ReachableWorkspaceCfg:
      - goal_ee: widen the lateral (pos_y) range and slightly extend pos_l so the
        EE target requires the whole body (incl. legs) to reposition, while
        keeping x_offset=0 and a conservative pitch range so the arm stays
        reachable (b2_z1's arm base is at the body center).
      - rewards: relax the default-pose bias (walking_dof), ask for slightly more
        decisive forward tracking (tracking_lin_vel_max), and re-enable the front
        leg lift penalty (feet_height) without introducing Radar-only rewards.
    """

    class goal_ee(B2Z1ReachableWorkspaceCfg.goal_ee):
        class sphere_center(B2Z1ReachableWorkspaceCfg.goal_ee.sphere_center):
            x_offset = 0.0
            z_invariant_offset = 0.8

        class ranges(B2Z1ReachableWorkspaceCfg.goal_ee.ranges):
            pos_l = [0.45, 0.82]
            pos_p = [0.10, 0.55]
            pos_y = [-1.2, 1.2]

    class rewards(B2Z1ReachableWorkspaceCfg.rewards):
        feet_height_target = 0.36
        class scales(B2Z1ReachableWorkspaceCfg.rewards.scales):
            # Relax the default-pose bias and ask for slightly more decisive
            # forward tracking without using the Radar-only rewards.
            walking_dof = 1.0
            tracking_lin_vel_max = 2.5
            feet_height = 0.5
            feet_drag = -0.10

        class arm_scales(B2Z1ReachableWorkspaceCfg.rewards.arm_scales):
            tracking_ee_world = 1.2


class B2Z1ReachableWorkspaceMotionCfgPPO(B2Z1ReachableWorkspaceCfgPPO):
    class algorithm(B2Z1ReachableWorkspaceCfgPPO.algorithm):
        # Full-weight arm advantage mixed in from step 0, ramping to 1.0 over
        # 3000 iters. This lets the EE target drive the whole body (incl. legs),
        # which is the main driver behind the observed leg motion.
        mixing_schedule = [1.0, 0, 3000]

    class runner(B2Z1ReachableWorkspaceCfgPPO.runner):
        experiment_name = 'b2_z1_reachable_workspace_motion'
        save_interval = 500


class B2Z1ReachableWorkspaceMotionPlusCfg(B2Z1ReachableWorkspaceMotionCfg):
    """Conservative-gait variant with a farther, upward-biased arm workspace.

    Compared to the previous Motion settings:
      - goal_ee: extend pos_l (0.88 -> 0.95) so targets sit at the edge of the
        arm+sphere workspace and require real repositioning, and shift pos_p
        upward ([-1.00, 0.80] -> [-0.60, 0.85]) to sample more high targets
        instead of deep downward reaches. The (l, p) corner is kept within
        ~1.1m of the arm base (~0.57m height + 0.7m reach) so most targets
        stay reachable.
      - rewards: back to conservative locomotion — walking_dof back to 1.5
        (B1 default pose bias) and tracking_lin_vel_max back to 2.0 to avoid
        the body-pitch-instead-of-stepping exploit; feet_height kept at 1.0
        with a higher lift target (0.30 -> 0.36) and a stronger drag penalty
        (-0.08 -> -0.10) to clean up shuffling.
    """

    class goal_ee(B2Z1ReachableWorkspaceMotionCfg.goal_ee):
        class sphere_center(B2Z1ReachableWorkspaceMotionCfg.goal_ee.sphere_center):
            x_offset = 0.0
            z_invariant_offset = 0.8

        class ranges(B2Z1ReachableWorkspaceMotionCfg.goal_ee.ranges):
            pos_l = [0.45, 0.82]
            pos_p = [0.10, 0.55]
            pos_y = [-1.2, 1.2]

    class rewards(B2Z1ReachableWorkspaceMotionCfg.rewards):
        feet_height_target = 0.36

        class scales(B2Z1ReachableWorkspaceMotionCfg.rewards.scales):
            walking_dof = 1.5
            tracking_lin_vel_max = 3.0
            feet_height = 1.0
            feet_drag = -0.10

        class arm_scales(B2Z1ReachableWorkspaceCfg.rewards.arm_scales):
            tracking_ee_world = 1.2


class B2Z1ReachableWorkspaceMotionPlusCfgPPO(B2Z1ReachableWorkspaceMotionCfgPPO):
    class runner(B2Z1ReachableWorkspaceMotionCfgPPO.runner):
        experiment_name = 'b2_z1_reachable_workspace_motion_plus'
        save_interval = 500
