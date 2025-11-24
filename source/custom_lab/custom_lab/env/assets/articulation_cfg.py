# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.utils import configclass

from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from .articulation import ArticulationWithCM


@configclass
class ArticulationCfgCM(ArticulationCfg):
    class_type: type = ArticulationWithCM
