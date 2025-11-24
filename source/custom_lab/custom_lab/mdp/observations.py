from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from custom_lab.env.assets import ArticulationWithCM

if TYPE_CHECKING:
    from custom_lab.env.g1_env import G1ModularEnv


def gait_phase(env: G1ModularEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def centroidal_angular_momentum_mixed(env: G1ModularEnv) -> torch.Tensor:
    """The centroidal angular momentum of the asset in the mixed frame. \
       x, y axis in the base frame, z axis in the world frame."""
    if not hasattr(env, "CM"):
        env.CM = torch.zeros(env.num_envs, 6, dtype=torch.float, device=env.device, requires_grad=False)
    if not hasattr(env, "CM_bf"):
        env.CM_bf = torch.zeros(env.num_envs, 6, dtype=torch.float, device=env.device, requires_grad=False)
    return torch.hstack([env.CM_bf[:, 3:5], env.CM[:, 5:6]])


def centroidal_angular_momentum_des(env: G1ModularEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    if not hasattr(env, "CM_des"):
        env.CM_des = torch.zeros(env.num_envs, 6, dtype=torch.float, device=env.device, requires_grad=False)
    return env.CM_des[:, 3:]


def centroidal_angular_momentum_des_bf(env: G1ModularEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset in the base frame."""
    if not hasattr(env, "CM_des_bf"):
        env.CM_des_bf = torch.zeros(env.num_envs, 6, dtype=torch.float, device=env.device, requires_grad=False)
    return env.CM_des_bf[:, 3:]


def centroidal_angular_momentum_des_mixed(env: G1ModularEnv) -> torch.Tensor:
    """The desired centroidal angular momentum of the asset in the mixed frame. \
       x, y axis in the base frame, z axis in the world frame."""
    if not hasattr(env, "CM_des"):
        env.CM_des = torch.zeros(env.num_envs, 6, dtype=torch.float, device=env.device, requires_grad=False)
    if not hasattr(env, "CM_des_bf"):
        env.CM_des_bf = torch.zeros(env.num_envs, 6, dtype=torch.float, device=env.device, requires_grad=False)
    return torch.hstack([env.CM_des_bf[:, 3:5], env.CM_des[:, 5:6]])


def centroidal_yaw_momentum(env: G1ModularEnv) -> torch.Tensor:
    """The centroidal momentum of the asset."""
    if not hasattr(env, "CM"):
        env.CM = torch.zeros(env.num_envs, 6, dtype=torch.float, device=env.device, requires_grad=False)
    return env.CM[:, 5:6]


def centroidal_yaw_momentum_des(env: G1ModularEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    if not hasattr(env, "CM_des"):
        env.CM_des = torch.zeros(env.num_envs, 6, dtype=torch.float, device=env.device, requires_grad=False)
    return env.CM_des[:, 5:6]


def CoM_pos_relative_to_feet(env: G1ModularEnv) -> torch.Tensor:
    """The CoM position relative to the center of the feet in the base frame."""
    if not hasattr(env, "CoM"):
        env.CoM = torch.zeros(env.num_envs, 3, dtype=torch.float, device=env.device, requires_grad=False)  # Center of Mass
    if not hasattr(env, "foot_states"):
        env.robot = env.scene["robot"]
        env.feet_ids = [
            env.robot.data.body_names.index("right_ankle_roll_link"),
            env.robot.data.body_names.index("left_ankle_roll_link"),
        ]
        env.foot_states = torch.zeros(env.num_envs, len(env.feet_ids), 7, dtype=torch.float, device=env.device, requires_grad=False)
    com_xy = env.CoM[:, :2]
    lf_pos = env.foot_states[:, 1, :2]
    rf_pos = env.foot_states[:, 0, :2]
    return com_xy - 0.5 * (lf_pos + rf_pos)
