import sys
import os,shutil
import time
import numpy as np
sys.path.insert(0, "/opt/openrobots/lib/python3.10/site-packages")
from pinocchio import casadi as cpin
import pinocchio as pin
import casadi as ca
from casadi import Function, SX
import cusadi.run_codegen as cu
import ctypes
Key_point_joint_names = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    # "left_hip_pitch_joint",
    # "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

    