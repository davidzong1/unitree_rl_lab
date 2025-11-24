from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass
from .step_command import StepCommand
from isaaclab.managers import CommandTermCfg


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class StepCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = StepCommand

    step_period: float = MISSING
    """Step period [env.step_dt]."""

    dstep_length: float | None = None
    """Desired step length [m]."""

    dstep_width: float | None = None
    """Desired step width [m]."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the step commands."""

        lin_vel_x: tuple[float, float] = MISSING  # min max [m/s]
        lin_vel_y: tuple[float, float] = MISSING  # min max [m/s]
        ang_vel_z: tuple[float, float] = MISSING  # min max [rad/s]
        heading: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges | None = None
    """Distribution ranges for the velocity commands."""
