from __future__ import annotations
import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar
from isaaclab.assets import Articulation
from custom_lab.env.assets import ArticulationWithCM
from isaaclab.managers import (
    CommandManager,
    CurriculumManager,
    RewardManager,
    TerminationManager,
)
from isaaclab.envs import ManagerBasedEnv, VecEnvObs, ManagerBasedRLEnvCfg
from custom_lab.isaaclab.manager_base_rl_env import ManagerBasedRLEnv_CM
from labDynamics.forward_kinematics import (
    PINOCCHIO_CASADI_FUNCTIONS_DIR,
    CUSADI_ROOT_DIR,
)
from labDynamics.key_point import Key_point_joint_names
import sys

sys.path.insert(0, CUSADI_ROOT_DIR)
from casadi import Function
from cusadi import *
import isaaclab.utils.math as math_utils
from labDynamics.pin_lab_convert import PinLabConvert
from custom_lab.Agent.task.g1_task_cfg import RobotSceneCfg

G1_LEG_JOINT_SORT: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
G1_WAIST_JOINT_SORT: list[str] = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
G1_ARM_JOINT_SORT: list[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
G1_JOINT_SORT: list[str] = G1_LEG_JOINT_SORT + G1_WAIST_JOINT_SORT + G1_ARM_JOINT_SORT
URDF_JOINT_NAME = G1_JOINT_SORT
CONTACT_LENGTH_NUM = 4  # 接触长度序列
command_vel_threshold = 0.1
desire_init_pos: dict[str, float] = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.28,
    "left_shoulder_roll_joint": 0.3,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.8,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.28,
    "right_shoulder_roll_joint": -0.3,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.8,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


class G1ModularEnv(ManagerBasedRLEnv_CM):
    waist_joints_name = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    leg_joints_name = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    arm_joints_name = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    num_base_joints: int = 6  # Floating base joint
    num_waist_joints = len(waist_joints_name)
    num_leg_joints = len(leg_joints_name)
    num_arm_joints = len(arm_joints_name)
    num_total_joints = num_waist_joints + num_arm_joints + num_leg_joints
    # 机器人总维度=基座+腿部+腰部+手部/
    num_total_dim = num_base_joints + num_leg_joints + num_waist_joints + num_arm_joints
    urdf_name = "g1_29dof"

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # 初始化父类
        super().__init__(cfg, render_mode, **kwargs)
        self.robot: ArticulationWithCM = self.scene["robot"]
        self.contact_sensor = self.scene.sensors["contact_forces"]
        self.convert = PinLabConvert(self.urdf_name, G1_JOINT_SORT, self.robot.joint_names)
        # 初始化固定参数
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.body_feet_ids = [
            self.robot.data.body_names.index("right_ankle_roll_link"),
            self.robot.data.body_names.index("left_ankle_roll_link"),
        ]
        self.feet_ids = [
            self.robot.data.body_names.index("right_ankle_roll_link"),
            self.robot.data.body_names.index("left_ankle_roll_link"),
        ]

        self.phase_sin = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_cos = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_schedule = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # num_envs x (left & right foot) x (x, y, z, quat)
        self.foot_states = torch.zeros(self.num_envs, len(self.feet_ids), 7, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_velocity = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (left & right foot) x (x, y, z)
        self.toe_heel_pos = torch.zeros(
            self.num_envs,
            2 * len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (right toe, heel / left toe, heel) x (x, y, z)
        self.foot_heading = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # num_envs x (left & right foot heading)
        self.foot_height_vec = torch.tensor([0.0, 0.0, -0.03]).repeat(self.num_envs, 1).to(self.device)
        self.foot_to_toe_vec = torch.tensor([0.105, 0.0, 0.0]).repeat(self.num_envs, 1).to(self.device)
        self.foot_to_heel_vec = torch.tensor([-0.045, 0.0, 0.0]).repeat(self.num_envs, 1).to(self.device)
        self.vel_command = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)  # (v_x, v_y, w_z) wrt base frame
        # Centroidal Momentum related
        self.CoM = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)  # Center of Mass
        self.CM = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)  # Centroidal Momentum
        self.CM_base = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Contribution of base joint to Centroidal Momentum
        self.CM_leg = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Contribution of leg joints to Centroidal Momentum
        self.CM_waist = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Contribution of waist joints to Centroidal Momentum
        self.CM_arm = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Contribution of arm joints to Centroidal Momentum
        self.CM_des = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)  # Desired Centroidal Momentum

        self.dCM = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)  # Time derivative of Centroidal Momentum
        self.dCM_base = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Contribution of base joint to the time derivative of Centroidal Momentum
        self.dCM_leg = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Contribution of leg joints to the time derivative of Centroidal Momentum
        self.dCM_waist = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Contribution of waist joints to the time derivative of Centroidal Momentum
        self.dCM_arm = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Contribution of arm joints to the time derivative of Centroidal Momentum
        self.dCM_des = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Desired time derivative of Centroidal Momentum

        self.rot_matrix_block = torch.zeros(self.num_envs, 6, 6, device=self.device, requires_grad=False)

        self.CM_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)  # Centroidal Momentum in base frame
        self.CM_base_bf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )  # Contribution of base joint to Centroidal Momentum in base frame
        self.CM_leg_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.CM_waist_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.CM_arm_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.CM_des_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)

        self.dCM_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.dCM_base_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.dCM_leg_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.dCM_waist_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.dCM_arm_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.dCM_des_bf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        # mimic 相关
        self.mimic_joint_pos = []
        self.mimic_joint_vel = []
        self.mimic_joint_pos_des = []
        self.mimic_joint_vel_des = []
        for i in range(len(Key_point_joint_names)):
            self.mimic_joint_pos.append(
                torch.zeros(
                    self.num_envs,
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
            )
            self.mimic_joint_vel.append(
                torch.zeros(
                    self.num_envs,
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
            )
            self.mimic_joint_pos_des.append(
                torch.zeros(
                    self.num_envs,
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
            )
            self.mimic_joint_vel_des.append(
                torch.zeros(
                    self.num_envs,
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
            )

        # * Observation variables
        self.step_period = self.cfg.commands.step_command.step_period * torch.ones(
            self.num_envs, 1, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.full_step_period = 2 * self.step_period.clone()

        self.current_step = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # (left & right foot) x (x, y, heading) wrt base x,y-coordinate
        self.num_base_joints = 6  # Floating base joint
        self.num_waist_joints = len(self.waist_joints_name)
        self.num_leg_joints = len(self.leg_joints_name)
        self.num_arm_joints = len(self.arm_joints_name)
        self.num_total_joints = self.num_waist_joints + self.num_arm_joints + self.num_leg_joints
        # 机器人总维度=基座+腿部+腰部+手部
        self.num_total_dim = self.num_base_joints + self.num_total_joints  # 6 + num_joint

        self.leg_joints_idx = [self.robot.data.joint_names.index(joint) for joint in self.leg_joints_name]
        self.waist_joints_idx = [self.robot.data.joint_names.index(joint) for joint in self.waist_joints_name]
        self.arm_joints_idx = [self.robot.data.joint_names.index(joint) for joint in self.arm_joints_name]
        self.foot_joint_idx = [
            self.robot.data.joint_names.index("left_ankle_roll_joint"),
            self.robot.data.joint_names.index("right_ankle_roll_joint"),
        ]

        self.gen_coord_pin = torch.zeros(
            self.num_envs,
            1 + self.num_total_dim,
            device=self.device,
            requires_grad=False,
        )
        self.gen_vel_body_pin = torch.zeros(self.num_envs, self.num_total_dim, device=self.device, requires_grad=False)
        self.gen_vel_body_pin_des = torch.zeros(self.num_envs, self.num_total_dim, device=self.device, requires_grad=False)
        self.gen_coord_pin_RS = torch.zeros(
            self.num_envs, self.num_total_dim, device=self.device, requires_grad=False
        )  # For RS, (roll, pitch, yaw) instead of (quaternion)
        # contact data
        self.contact_length_b = torch.zeros(
            self.num_envs,
            CONTACT_LENGTH_NUM,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # contact length history for left & right foot
        self.old_contact_flag = torch.zeros(
            self.num_envs,
            len(self.feet_ids),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.old_contact_flag[:, :] = True  # 初始化时双脚均接触地面
        self.cur_contact_flag = self.old_contact_flag.clone()
        # 配置urdf_name变量
        default_joint_pos_as_urdf_type = [0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0]
        for joint_name in URDF_JOINT_NAME:
            default_joint_pos_as_urdf_type.append(desire_init_pos[joint_name])
        # create a (num_envs x len(default_joint_pos_as_urdf_type)) tensor of default joint positions
        q_default = (
            torch.tensor(
                default_joint_pos_as_urdf_type,
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        # 加载CasADi函数
        self.key_point_pos_fn_array = []
        for i, joint_name in enumerate(Key_point_joint_names):
            key_point_pos_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{joint_name}_pos_bf_{self.urdf_name}.casadi")
            self.key_point_pos_fn_array.append(CusadiFunction(key_point_pos_fn, num_instances=self.num_envs, precision="float"))
            fn = self.key_point_pos_fn_array[i]
            fn.evaluate(q_default.to(self.device))
            self.mimic_joint_pos_des[i] = fn.outputs_sparse[0].to(self.device, dtype=torch.float32).reshape_as(self.mimic_joint_pos[i])

        self.key_point_vel_fn_array = []
        for joint_name in Key_point_joint_names:
            joint_vel_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{joint_name}_lin_vel_{self.urdf_name}.casadi")
            self.key_point_vel_fn_array.append(CusadiFunction(joint_vel_fn, num_instances=self.num_envs, precision="float"))

        self.M_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/M_{self.urdf_name}.casadi")  # Mass matrix
        self.CMM_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CMM_{self.urdf_name}.casadi")  # Centroidal Momentum Matrix
        self.dCMM_fn = Function.load(
            f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/dCMM_{self.urdf_name}.casadi"
        )  # Time derivative of Centroidal Momentum Matrix
        self.CM_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CM_{self.urdf_name}.casadi")  # Centroidal Momentum
        self.dCM_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/dCM_{self.urdf_name}.casadi")  # Time derivative of Centroidal Momentum
        self.CoM_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CoM_{self.urdf_name}.casadi")  # Center of Mass
        # self.lf_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/lf_{self.urdf_name}.casadi") # Left Foot
        self.base_pos_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/base_pos_{self.urdf_name}.casadi")  # Base position
        self.base_rot_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/base_rot_{self.urdf_name}.casadi")  # Base orientation in SO(3)

        self.cusadi_CMM_fn = CusadiFunction(self.CMM_fn, num_instances=self.num_envs, precision="float")
        self.cusadi_dCMM_fn = CusadiFunction(self.dCMM_fn, num_instances=self.num_envs, precision="float")
        self.cusadi_CM_fn = CusadiFunction(self.CM_fn, num_instances=self.num_envs, precision="float")
        self.cusadi_dCM_fn = CusadiFunction(self.dCM_fn, num_instances=self.num_envs, precision="float")
        self.cusadi_CoM_fn = CusadiFunction(self.CoM_fn, num_instances=self.num_envs, precision="float")

        self.action = torch.zeros(self.num_envs, self.num_total_joints, device=self.device)

    def _post_physics_step_callback(self):
        self.progress_within_step = self.episode_length_buf.unsqueeze(1) % self.full_step_period
        self.phase = self.progress_within_step / self.full_step_period

        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.contact_forces = self.contact_sensor.data.net_forces_w

        self.foot_contact = torch.gt(self.contact_forces[:, self.feet_ids, 2], 0)

        self.contact_schedule = self.smooth_sqr_wave(self.phase)

        self.vel_command = self.command_manager.get_command("base_velocity")

        self._compute_generalized_coordinates()

        # self._compute_key_point_positions()

        # self._compute_key_point_velocities()

        self._calculate_CoM()

        self._calculate_centroidal_momentum()

        self._calculate_foot_state()
        # self._calculate_contact_length()
        pass

    def _calculate_foot_state(self):
        """Calculate the foot velocity in world frame [ lf, rf ]"""
        rfoot_state = self.robot.data.body_state_w[:, self.body_feet_ids[0], :7]
        lfoot_state = self.robot.data.body_state_w[:, self.body_feet_ids[1], :7]
        rfoot_height_vec_in_world = math_utils.quat_apply(rfoot_state[:, 3:7], self.foot_height_vec)
        rfoot_to_toe_vec_in_world = math_utils.quat_apply(rfoot_state[:, 3:7], self.foot_to_toe_vec)
        rfoot_to_heel_vec_in_world = math_utils.quat_apply(rfoot_state[:, 3:7], self.foot_to_heel_vec)
        lfoot_height_vec_in_world = math_utils.quat_apply(lfoot_state[:, 3:7], self.foot_height_vec)
        lfoot_to_toe_vec_in_world = math_utils.quat_apply(lfoot_state[:, 3:7], self.foot_to_toe_vec)
        lfoot_to_heel_vec_in_world = math_utils.quat_apply(lfoot_state[:, 3:7], self.foot_to_heel_vec)
        # 加上鞋底高度偏移
        self.foot_states[:, 0, :3] = lfoot_state[:, :3] + lfoot_height_vec_in_world
        self.foot_states[:, 0, 3:7] = lfoot_state[:, 3:7]
        self.foot_states[:, 1, :3] = rfoot_state[:, :3] + rfoot_height_vec_in_world
        self.foot_states[:, 1, 3:7] = rfoot_state[:, 3:7]
        self.toe_heel_pos[:, 0, :] = self.foot_states[:, 0, :3] + rfoot_to_toe_vec_in_world
        self.toe_heel_pos[:, 1, :] = self.foot_states[:, 0, :3] + rfoot_to_heel_vec_in_world
        self.toe_heel_pos[:, 2, :] = self.foot_states[:, 1, :3] + lfoot_to_toe_vec_in_world
        self.toe_heel_pos[:, 3, :] = self.foot_states[:, 1, :3] + lfoot_to_heel_vec_in_world
        self.foot_velocity[:, 0] = self.robot.data.body_state_w[:, self.body_feet_ids[1], 7:10]
        self.foot_velocity[:, 1] = self.robot.data.body_state_w[:, self.body_feet_ids[0], 7:10]

    def _calculate_contact_length(self):
        self.cur_contact_flag[:, 0] = self.contact_sensor.data.net_forces_w[:, 1, 2] > 1.0
        self.cur_contact_flag[:, 1] = self.contact_sensor.data.net_forces_w[:, 0, 2] > 1.0
        # double change(float to contact), left change, right change,no change
        left_changed = self.cur_contact_flag[:, 0] != self.old_contact_flag[:, 0]
        right_changed = self.cur_contact_flag[:, 1] != self.old_contact_flag[:, 1]
        turn_foot = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        mask_double = left_changed & right_changed
        mask_left = left_changed & ~right_changed
        mask_right = ~left_changed & right_changed
        turn_foot[mask_double] = 0
        turn_foot[mask_left] = 1
        turn_foot[mask_right] = 2
        if torch.all(turn_foot == -1):
            return
        self.contact_length_b[mask_double, 1:] = self.contact_length_b[mask_double, :-1]
        self.contact_length_b[mask_left, 1:] = self.contact_length_b[mask_left, :-1]
        self.contact_length_b[mask_right, 1:] = self.contact_length_b[mask_right, :-1]
        # compare max distance from foot to root in float
        self.contact_length_b[mask_double, 0] = torch.max(
            (self.contact_sensor.data.pos_w[mask_double, 0, :2] - self.robot.data.root_pos_w[mask_double, :2]).norm(dim=1),
            (self.contact_sensor.data.pos_w[mask_double, 1, :2] - self.robot.data.root_pos_w[mask_double, :2]).norm(dim=1),
        )
        self.contact_length_b[mask_left, 0] = (self.contact_sensor.data.pos_w[mask_left, 1, :2] - self.robot.data.root_pos_w[mask_left, :2]).norm(
            dim=-1
        )
        self.contact_length_b[mask_right, 0] = (self.contact_sensor.data.pos_w[mask_right, 0, :2] - self.robot.data.root_pos_w[mask_right, :2]).norm(
            dim=-1
        )

        self.old_contact_flag = self.cur_contact_flag.clone()

    def _compute_key_point_positions(self):
        """Compute key point positions for mimic task"""
        for i, fn in enumerate(self.key_point_pos_fn_array):
            fn.evaluate(self.gen_coord_pin.to(self.device))
            self.mimic_joint_pos[i] = fn.outputs_sparse[0].to(self.device, dtype=torch.float32).reshape_as(self.mimic_joint_pos[i])
        pass

    def _compute_key_point_velocities(self):
        """Compute key point velocities for mimic task"""
        for i, fn in enumerate(self.key_point_vel_fn_array):
            fn.evaluate(
                [
                    self.gen_coord_pin.to(self.device),
                    self.gen_vel_body_pin.to(self.device),
                ]
            )
            self.mimic_joint_vel[i] = fn.outputs_sparse[0].to(self.device, dtype=torch.float32).reshape_as(self.mimic_joint_vel[i])
        pass

    def _compute_generalized_coordinates(self):
        """Compute generalized coordinates and generalized velocities"""
        gen_coord = torch.hstack(
            (
                self.robot.data.root_link_pos_w,
                self.robot.data.root_link_quat_w,
                self.robot.data.joint_pos,
            )
        )
        self.gen_coord_pin = self.convert.q_lab_to_pin(gen_coord)
        gen_vel_body = torch.hstack(
            (
                self.robot.data.root_link_lin_vel_b,
                self.robot.data.root_link_ang_vel_b,
                self.robot.data.joint_vel,
            )
        )
        self.gen_vel_body_pin = self.convert.v_lab_to_pin(gen_vel_body)
        gen_acc_body = torch.hstack(
            (
                self.robot.data.root_link_lin_acc_b,
                self.robot.data.root_link_ang_acc_b,
                self.robot.data.joint_acc,
            )
        )
        self.gen_acc_body_pin = self.convert.v_lab_to_pin(gen_acc_body)

        self.gen_vel_body_pin_des[:, 0] = self.vel_command[:, 0]
        self.gen_vel_body_pin_des[:, 1] = self.vel_command[:, 1]
        self.gen_vel_body_pin_des[:, 5] = self.vel_command[:, 2]

        self.gen_coord_pin_RS = torch.hstack(
            (
                self.robot.data.root_link_pos_w,
                torch.stack(
                    math_utils.euler_xyz_from_quat(self.robot.data.root_link_quat_w),
                    dim=1,
                ),
                self.robot.data.joint_pos,
            )
        )
        self.gen_coord_pin_RS = self.convert.v_lab_to_pin(self.gen_coord_pin_RS)

    def _calculate_CoM(self):
        """Calculate the CoM position of the robot"""
        self.cusadi_CoM_fn.evaluate([self.gen_coord_pin.to("cuda")])
        self.CoM = self.cusadi_CoM_fn.outputs_sparse[0].to(self.device).float()

    def _calculate_centroidal_momentum(self):
        """Calculate the centroidal momelntum of the robot
        IsaacLab's mass matrix is projected to the root link's CoM frame
        We need to convert it to whole body's CoM frame
        Convert from root_link_CoM (world frame) -> root_link (base frame) -> CoM (world frame)
        """
        """ CMM函数flow_map """
        self.cusadi_CMM_fn.evaluate([self.gen_coord_pin.to("cuda")])
        CMM = self.cusadi_CMM_fn.outputs_sparse[0].reshape(self.num_envs, -1, self.num_base_joints).permute(0, 2, 1).to(self.device)
        """ dCMM函数flow_map """
        self.cusadi_dCMM_fn.evaluate([self.gen_coord_pin.to("cuda"), self.gen_vel_body_pin.to("cuda")])
        dCMM = self.cusadi_dCMM_fn.outputs_sparse[0].reshape(self.num_envs, -1, self.num_base_joints).permute(0, 2, 1).to(self.device)

        self.CM = (CMM @ self.gen_vel_body_pin.unsqueeze(2)).squeeze(2).float()
        self.CM_base = (
            (CMM[:, :, : self.num_base_joints] @ self.gen_vel_body_pin[:, : self.num_base_joints].unsqueeze(2)).squeeze(2).float()
        )  # A(q) * qdot
        self.CM_leg = (
            (
                CMM[
                    :,
                    :,
                    self.num_base_joints : self.num_base_joints + self.num_leg_joints,
                ]
                @ self.gen_vel_body_pin[:, self.num_base_joints : self.num_base_joints + self.num_leg_joints].unsqueeze(2)
            )
            .squeeze(2)
            .float()
        )  # A(q) * qdot
        self.CM_waist = (
            (
                CMM[
                    :,
                    :,
                    self.num_base_joints + self.num_leg_joints : self.num_base_joints + self.num_leg_joints + self.num_waist_joints,
                ]
                @ self.gen_vel_body_pin[
                    :,
                    self.num_base_joints + self.num_leg_joints : self.num_base_joints + self.num_leg_joints + self.num_waist_joints,
                ].unsqueeze(2)
            )
            .squeeze(2)
            .float()
        )
        self.CM_arm = (
            (
                CMM[
                    :,
                    :,
                    (self.num_base_joints + self.num_leg_joints + self.num_waist_joints) :,
                ]
                @ self.gen_vel_body_pin[
                    :,
                    (self.num_base_joints + self.num_leg_joints + self.num_waist_joints) :,
                ].unsqueeze(2)
            )
            .squeeze(2)
            .float()
        )  # A(q) * qdot
        self.CM_des = (CMM @ self.gen_vel_body_pin_des.unsqueeze(2)).squeeze(2).float()

        self.dCM = (dCMM @ self.gen_vel_body_pin.unsqueeze(2)).squeeze(2).float() + (CMM @ self.gen_acc_body_pin.unsqueeze(2)).squeeze(
            2
        ).float()  # dCM = dA(q) * qdot + A(q) * qddot
        self.dCM_base = (dCMM[:, :, : self.num_base_joints] @ self.gen_vel_body_pin[:, : self.num_base_joints].unsqueeze(2)).squeeze(2).float() + (
            CMM[:, :, : self.num_base_joints] @ self.gen_acc_body_pin[:, : self.num_base_joints].unsqueeze(2)
        ).squeeze(2).float()
        self.dCM_leg = (
            dCMM[:, :, self.num_base_joints : self.num_base_joints + self.num_leg_joints]
            @ self.gen_vel_body_pin[:, self.num_base_joints : self.num_base_joints + self.num_leg_joints].unsqueeze(2)
        ).squeeze(2).float() + (
            CMM[:, :, self.num_base_joints : self.num_base_joints + self.num_leg_joints]
            @ self.gen_acc_body_pin[:, self.num_base_joints : self.num_base_joints + self.num_leg_joints].unsqueeze(2)
        ).squeeze(
            2
        ).float()
        self.dCM_waist = (
            dCMM[
                :,
                :,
                self.num_base_joints + self.num_leg_joints : self.num_base_joints + self.num_leg_joints + self.num_waist_joints,
            ]
            @ self.gen_vel_body_pin[
                :,
                self.num_base_joints + self.num_leg_joints : self.num_base_joints + self.num_leg_joints + self.num_waist_joints,
            ].unsqueeze(2)
        ).squeeze(2).float() + (
            CMM[
                :,
                :,
                self.num_base_joints + self.num_leg_joints : self.num_base_joints + self.num_leg_joints + self.num_waist_joints,
            ]
            @ self.gen_acc_body_pin[
                :,
                self.num_base_joints + self.num_leg_joints : self.num_base_joints + self.num_leg_joints + self.num_waist_joints,
            ].unsqueeze(2)
        ).squeeze(
            2
        ).float()
        self.dCM_arm = (
            dCMM[
                :,
                :,
                (self.num_base_joints + self.num_leg_joints + self.num_waist_joints) :,
            ]
            @ self.gen_vel_body_pin[
                :,
                (self.num_base_joints + self.num_leg_joints + self.num_waist_joints) :,
            ].unsqueeze(2)
        ).squeeze(2).float() + (
            CMM[
                :,
                :,
                (self.num_base_joints + self.num_leg_joints + self.num_waist_joints) :,
            ]
            @ self.gen_acc_body_pin[
                :,
                (self.num_base_joints + self.num_leg_joints + self.num_waist_joints) :,
            ].unsqueeze(2)
        ).squeeze(
            2
        ).float()

        rot_matrix = math_utils.matrix_from_quat(self.robot.data.root_link_quat_w).permute(0, 2, 1)  # Rotation matrix from world frame to base frame
        self.rot_matrix_block[:, :3, :3] = rot_matrix
        self.rot_matrix_block[:, 3:, 3:] = rot_matrix

        self.CM_bf = (self.rot_matrix_block @ self.CM.unsqueeze(2)).squeeze(2)
        self.CM_base_bf = (self.rot_matrix_block @ self.CM_base.unsqueeze(2)).squeeze(2)
        self.CM_leg_bf = (self.rot_matrix_block @ self.CM_leg.unsqueeze(2)).squeeze(2)
        self.CM_arm_bf = (self.rot_matrix_block @ self.CM_arm.unsqueeze(2)).squeeze(2)
        self.CM_des_bf = (self.rot_matrix_block @ self.CM_des.unsqueeze(2)).squeeze(2)

        self.dCM_bf = (self.rot_matrix_block @ self.dCM.unsqueeze(2)).squeeze(2)
        self.dCM_base_bf = (self.rot_matrix_block @ self.dCM_base.unsqueeze(2)).squeeze(2)
        self.dCM_leg_bf = (self.rot_matrix_block @ self.dCM_leg.unsqueeze(2)).squeeze(2)
        self.dCM_waist_bf = (self.rot_matrix_block @ self.dCM_waist.unsqueeze(2)).squeeze(2)
        self.dCM_arm_bf = (self.rot_matrix_block @ self.dCM_arm.unsqueeze(2)).squeeze(2)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def smooth_sqr_wave(self, phase, eps=1.0):
        p = 2.0 * torch.pi * phase
        return torch.sin(p) / torch.sqrt(torch.sin(p) ** 2.0 + eps**2.0)

    def smooth_sqr_wave_advanced(self, phase, eps=0.2, a=0.2):
        """Advanced contact scheduler containing the flying phase
        a: fraction of the flying phase
        """
        t = (1 - a) / 2

        # Piecewise function for the first half of the cycle (right foot)
        r = 2.0 * torch.pi * phase / (1 - a)
        r_output = torch.sin(r) / torch.sqrt(torch.sin(r) ** 2 + eps**2)

        # Piecewise function for the second half of the cycle (left foot)
        l = 2.0 * torch.pi * (phase - (0.5 - t)) / (1 - a)
        l_output = torch.sin(l) / torch.sqrt(torch.sin(l) ** 2 + eps**2)

        # Create masks
        mask_r = phase < t  # First part of gait cycle (right foot)
        mask_l = (0.5 <= phase) & (phase < 0.5 + t)  # Second part of gait cycle (left foot)

        # Apply masks to calculate the output
        output = r_output * mask_r + l_output * mask_l

        return output
