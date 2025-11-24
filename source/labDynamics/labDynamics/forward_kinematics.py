"""
Generates casadi functions for forward kinematics and jacobian computation using Pinocchio library.
https://github.com/stack-of-tasks/pinocchio
"""

import sys
import os, shutil
import time
import numpy as np

# sys.path.insert(0, "/opt/openrobots/lib/python3.10/site-packages")
from pinocchio import casadi as cpin
import pinocchio as pin
import casadi as ca
from casadi import Function, SX
import ctypes

LABDYNAMIC_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PINOCCHIO_CASADI_FUNCTIONS_DIR = os.path.join(LABDYNAMIC_ROOT_DIR, "casadi_functions")
DYNAMIC_ROOT_DIR = os.path.dirname(__file__)
CUSADI_ROOT_DIR = os.path.join(LABDYNAMIC_ROOT_DIR, "cusadi")
print("LABDYNAMIC_ROOT_DIR:", LABDYNAMIC_ROOT_DIR)
print("PINOCCHIO_CASADI_FUNCTIONS_DIR:", PINOCCHIO_CASADI_FUNCTIONS_DIR)
np.set_printoptions(threshold=1000, linewidth=1000, precision=6)
os.makedirs(PINOCCHIO_CASADI_FUNCTIONS_DIR, exist_ok=True)
print("Contents:", os.listdir(PINOCCHIO_CASADI_FUNCTIONS_DIR))
# Make the dynamics directory importable and import the key_point module
sys.path.insert(0, DYNAMIC_ROOT_DIR)
sys.path.insert(0, CUSADI_ROOT_DIR)
import labDynamics.key_point as key_point
import cusadi.run_codegen as cu

# 执行删除
for name in os.listdir(PINOCCHIO_CASADI_FUNCTIONS_DIR):
    path = os.path.join(PINOCCHIO_CASADI_FUNCTIONS_DIR, name)
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    except Exception as e:
        print("Failed to remove", path, e)
print("Cleared:", PINOCCHIO_CASADI_FUNCTIONS_DIR)


def test_forward_kinematics_from_urdf(urdf_path, frame_name=None):
    """
    Loads a URDF into a Pinocchio model, runs forward kinematics for a test joint config,
    and returns the final pose (position + rotation) of a given frame.

    Args:
        urdf_path  (str): The path to your robot's URDF file on disk.
        frame_name (str): Name of the frame (link) you want the pose for. If None, returns all frames.

    Returns:
        pos (3-array):    The position of the chosen frame in world coords (if frame_name is given).
        rot (3x3 array):  The rotation matrix of the chosen frame in world coords (if frame_name is given).
        data.oMf (list):  The list of all frames' transforms if frame_name is None.
    """

    ### * Pinocchio tutorial using numeric numpy array * ###
    print("==== Numeric Version ====")
    # Build the Pinocchio model from the URDF file
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    print("model name: " + model.name)
    for j in range(model.njoints):
        print(j, model.names[j], model.joints[j].nq, model.joints[j].nv)
    print()
    for j in range(model.nframes):
        print(j, model.frames[j].name)

    data = model.createData()

    # Sample a random configuration
    # q_num = pin.randomConfiguration(model)
    # q_num = pin.neutral(model)
    q_num = np.random.rand(model.nq)
    q_dot_num = np.random.rand(model.nv)
    q_num = np.zeros((int(model.nq)))
    q_num[2] = 0.79
    q_num[6] = 1.0
    q_dot_num = np.zeros((int(model.nv)))
    q_ddot_num = np.zeros((int(model.nv)))

    print(f"q_num: {q_num.T}")
    print(f"q_dot_num: {q_dot_num.T}")
    print(f"q_ddot_num: {q_ddot_num.T}")

    # Run forward kinematics
    # pin.forwardKinematics(model, data, q_num) # Updates data for the joint placements
    # pin.forwardKinematics(model, data, q_num, q_dot_num) # Updates data for the joint placement and spatial velocities
    pin.forwardKinematics(model, data, q_num, q_dot_num, q_ddot_num)  # Updates data for the joint placement, spatial velocities, and accelerations
    # Updates data.oMf[] with transforms of each frame in the model.
    pin.updateFramePlacements(model, data)

    # Otherwise, look up the chosen frame
    frame_id = model.getFrameId(frame_name)
    # The transform from world (root) to that frame
    oMf = data.oMf[frame_id]
    pos = oMf.translation
    rot = oMf.rotation
    print(f"Frame {frame_name} \nposition: {pos} \nrotation: \n{rot}")

    J = pin.computeFrameJacobian(model, data, q_num, frame_id)

    frame_velocity = pin.getFrameVelocity(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
    linear_velocity = frame_velocity.linear
    angular_velocity = frame_velocity.angular
    print(f"Frame {frame_name} \nlinear velocity: {linear_velocity} \nangular velocity: {angular_velocity}")

    base_frame_id = model.getFrameId("pelvis")
    # The base frame position and rotation
    base_oMf = data.oMf[base_frame_id]
    base_pos = base_oMf.translation
    base_rot = base_oMf.rotation
    print(f"Frame base \nposition: {base_pos} \nrotation: \n{base_rot}")

    # Compute whole body dynamics equation of motion
    # M(q) @ q_ddot + C(q, q_dot) @ q_dot + G(q) = tau + J_c.T @ F_c
    M = pin.crba(model, data, q_num)
    C = pin.computeCoriolisMatrix(model, data, q_num, q_dot_num)
    G = pin.computeGeneralizedGravity(model, data, q_num)
    print(f"M(q): \n{M}")
    print(f"C(q, q_dot): \n{C}")
    print(f"G(q): \n{G}")

    CMM = pin.computeCentroidalMap(model, data, q_num)
    CM1 = pin.computeCentroidalMomentum(model, data, q_num, q_dot_num)
    CM2 = pin.computeCentroidalMomentum(model, data)
    CoM = pin.centerOfMass(model, data, q_num)
    print(f"CMM(q): \n{CMM}")
    print(f"CM1(q, q_dot): \n{CM1}")
    print(f"CM2(): \n{CM2}")
    print(f"CM3: \n{CMM @ q_dot_num}")
    print(f"CoM(q): \n{CoM}")
    print(f"CoM(q): \n{base_pos + base_rot @ np.array([M[5,1], M[3,2], M[4,0]]) / 24.88855}")  # M is linear / angular / joint

    dCMM = pin.computeCentroidalMapTimeVariation(model, data, q_num, q_dot_num)
    dCM1 = pin.computeCentroidalMomentumTimeVariation(model, data)
    dCM2 = pin.computeCentroidalMomentumTimeVariation(model, data, q_num, q_dot_num, q_ddot_num)
    print(f"dCMM(q, q_dot): \n{dCMM}")
    print(f"dCM1(q, q_dot, q_ddot): \n{dCM1}")
    print(f"dCM2(q, q_dot, q_ddot): \n{dCM2}")
    print(f"dCM3(q, q_dot, q_ddot): \n{dCMM @ q_dot_num + CMM @ q_ddot_num}")

    adjoint = np.zeros((24, 24))
    adjoint[0:3, 0:3] = base_rot.T
    adjoint[0:3, 3:6] = pin.skew(CoM - base_pos) @ base_rot.T
    adjoint[3:6, 3:6] = base_rot.T

    ### * Pinocchio tutorial using casadi symbolic array * ###
    print("\n==== Casadi Symbolic Version ====")
    # Create casadi model and data
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    print("model name: " + model.name)

    q_sym = SX.sym("q_sym", model.nq)
    q_dot_sym = SX.sym("q_dot_sym", model.nv)
    print(f"q_sym: {q_sym.T}")
    print(f"q_dot_sym: {q_dot_sym.T}")

    # Run forward kinematics
    # cpin.forwardKinematics(cmodel, cdata, q_sym)
    cpin.forwardKinematics(cmodel, cdata, q_sym, q_dot_sym)
    cpin.updateFramePlacements(cmodel, cdata)
    # Otherwise, look up the chosen frame
    frame_id = cmodel.getFrameId(frame_name)
    # The transform from world (root) to that frame
    coMf = cdata.oMf[frame_id]
    cpos = coMf.translation
    crot = coMf.rotation

    cframe_velocity = cpin.getFrameVelocity(cmodel, cdata, frame_id, pin.LOCAL_WORLD_ALIGNED)
    clinear_velocity = cframe_velocity.linear
    cangular_velocity = cframe_velocity.angular

    # Save as casadi functions
    fk = Function("fk", [q_sym], [cpos, crot])
    fk.expand()
    fk.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/fk_{frame_name}.casadi")

    fk_load = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/fk_{frame_name}.casadi")
    pos, rot = fk_load(q_num)
    print(f"Frame {frame_name} \nposition: {pos} \nrotation: \n{rot}")
    os.remove(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/fk_{frame_name}.casadi")


def save_forward_kinematics(urdf_name, urdf_path, frame_names):
    """
    Saves the forward kinematics in casadi function for multiple frames.

    Args:
        urdf_path (str): The path to the robot's URDF file on disk.
        frame_names (list): List of frame names to save forward kinematics for.
    """
    # Build the Pinocchio model from the URDF file
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    print("model name: " + model.name)

    # Create casadi model and data
    cmodel = cpin.Model(model)
    for j in range(cmodel.njoints):
        print(j, cmodel.names[j], cmodel.joints[j].nq, cmodel.joints[j].nv)
    cdata = cmodel.createData()

    q_sym = SX.sym("q_sym", model.nq)
    q_dot_sym = SX.sym("q_dot_sym", model.nv)
    q_ddot_sym = SX.sym("q_ddot_sym", model.nv)
    # print(f"q_sym: {q_sym.T}")
    # print(f"q_dot_sym: {q_dot_sym.T}")
    # print(f"q_ddot_sym: {q_ddot_sym.T}")
    Key_point_joint_idx = [model.getJointId(name) for name in key_point.Key_point_joint_names]
    print("Key_point_joint_idx:", Key_point_joint_idx)

    # Run forward kinematics
    # cpin.forwardKinematics(cmodel, cdata, q_sym, q_dot_sym)
    cpin.forwardKinematics(cmodel, cdata, q_sym, q_dot_sym, q_ddot_sym)
    cpin.updateFramePlacements(cmodel, cdata)
    # Otherwise, look up the chosen frame
    base_frame_id = cmodel.getFrameId("pelvis")
    coMf = cdata.oMf[base_frame_id]
    # The base frame position and rotation
    cpos = coMf.translation
    crot = coMf.rotation
    cpos_fn = Function(f"base_pos_{urdf_name}", [q_sym], [cpos])
    crot_fn = Function(f"base_rot_{urdf_name}", [q_sym], [crot])
    cpos_fn.expand()
    crot_fn.expand()
    cpos_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/base_pos_{urdf_name}.casadi")
    crot_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/base_rot_{urdf_name}.casadi")
    # 计算关键点坐标
    key_point_pos_terms = []
    key_point_quat_terms = []
    key_point_lin_vel_terms = []
    key_point_ang_vel_terms = []
    for i, joint_id in enumerate(Key_point_joint_idx):
        coMf_kp = cdata.oMi[joint_id]
        joint_vel = cdata.v[joint_id]
        key_point_pos_terms.append(crot.T @ (coMf_kp.translation - cpos))
        key_point_quat_terms.append(crot.T @ coMf_kp.rotation)
        key_point_lin_vel_terms.append(joint_vel.linear)
        key_point_ang_vel_terms.append(joint_vel.angular)
        key_point_pos_bf_fn = Function(f"key_point_{key_point.Key_point_joint_names[i]}_pos_bf_{urdf_name}", [q_sym], [key_point_pos_terms[-1]])
        key_point_rot_bf_fn = Function(f"key_point_{key_point.Key_point_joint_names[i]}_rot_bf_{urdf_name}", [q_sym], [key_point_quat_terms[-1]])
        key_point_lin_vel_fn = Function(
            f"key_point_{key_point.Key_point_joint_names[i]}_lin_vel_{urdf_name}", [q_sym, q_dot_sym], [key_point_lin_vel_terms[-1]]
        )
        key_point_ang_vel_fn = Function(
            f"key_point_{key_point.Key_point_joint_names[i]}_ang_vel_{urdf_name}", [q_sym, q_dot_sym], [key_point_ang_vel_terms[-1]]
        )
        key_point_pos_bf_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{key_point.Key_point_joint_names[i]}_pos_bf_{urdf_name}.casadi")
        key_point_rot_bf_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{key_point.Key_point_joint_names[i]}_rot_bf_{urdf_name}.casadi")
        key_point_lin_vel_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{key_point.Key_point_joint_names[i]}_lin_vel_{urdf_name}.casadi")
        key_point_ang_vel_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{key_point.Key_point_joint_names[i]}_ang_vel_{urdf_name}.casadi")
    # 动力学计算
    M = cpin.crba(cmodel, cdata, q_sym)  # pin.Convention.WORLD / pin.Convention.LOCAL
    C = cpin.computeCoriolisMatrix(cmodel, cdata, q_sym, q_dot_sym) @ q_dot_sym
    G = cpin.computeGeneralizedGravity(cmodel, cdata, q_sym)
    M_fn = Function(f"M_{urdf_name}", [q_sym], [M])
    C_fn = Function(f"C_{urdf_name}", [q_sym, q_dot_sym], [C])
    G_fn = Function(f"G_{urdf_name}", [q_sym], [G])
    M_fn.expand()
    C_fn.expand()
    G_fn.expand()
    M_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/M_{urdf_name}.casadi")
    C_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/C_{urdf_name}.casadi")
    G_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/G_{urdf_name}.casadi")

    CMM = cpin.computeCentroidalMap(cmodel, cdata, q_sym)
    dCMM = cpin.computeCentroidalMapTimeVariation(cmodel, cdata, q_sym, q_dot_sym)
    CM = cpin.computeCentroidalMomentum(cmodel, cdata, q_sym, q_dot_sym)
    # 基座系下的质心动量
    dCM = cpin.computeCentroidalMomentumTimeVariation(cmodel, cdata, q_sym, q_dot_sym, q_ddot_sym)
    # 世界坐标系下的质心位置
    CoM = cpin.centerOfMass(cmodel, cdata, q_sym)
    CMM_fn = Function(f"CMM_{urdf_name}", [q_sym], [CMM])
    dCMM_fn = Function(f"dCMM_{urdf_name}", [q_sym, q_dot_sym], [dCMM])
    CM_fn = Function(f"CM_{urdf_name}", [q_sym, q_dot_sym], [CM.linear, CM.angular])
    dCM_fn = Function(f"dCM_{urdf_name}", [q_sym, q_dot_sym, q_ddot_sym], [dCM.linear, dCM.angular])
    CoM_fn = Function(f"CoM_{urdf_name}", [q_sym], [CoM])
    CMM_fn.expand()
    dCMM_fn.expand()
    CM_fn.expand()
    dCM_fn.expand()
    CoM_fn.expand()
    CMM_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CMM_{urdf_name}.casadi")
    dCMM_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/dCMM_{urdf_name}.casadi")
    CM_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CM_{urdf_name}.casadi")
    dCM_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/dCM_{urdf_name}.casadi")
    CoM_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CoM_{urdf_name}.casadi")

    for frame_name in frame_names:
        # Run forward kinematics
        # cpin.forwardKinematics(cmodel, cdata, q_sym)
        cpin.forwardKinematics(cmodel, cdata, q_sym, q_dot_sym)
        cpin.updateFramePlacements(cmodel, cdata)

        # Otherwise, look up the chosen frame
        frame_id = cmodel.getFrameId(frame_name)
        # The transform from world (root) to that frame
        coMf = cdata.oMf[frame_id]
        cpos = coMf.translation
        crot = coMf.rotation

        cframe_velocity = cpin.getFrameVelocity(cmodel, cdata, frame_id, pin.LOCAL_WORLD_ALIGNED)
        clinear_velocity = cframe_velocity.linear
        cangular_velocity = cframe_velocity.angular

        # Save as casadi functions
        fk_fn = Function(f"fk_{urdf_name}", [q_sym], [cpos, crot])
        fk_fn.expand()
        fk_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/fk_{frame_name}_{urdf_name}.casadi")

        # Compute Jacobian
        jacobian = cpin.computeFrameJacobian(
            cmodel, cdata, q_sym, frame_id, pin.LOCAL_WORLD_ALIGNED
        )  # pin.WORLD / pin.LOCAL / pin.LOCAL_WORLD_ALIGNED
        jacobian_fn = Function(f"jacobian_{frame_name}_{urdf_name}", [q_sym], [jacobian])
        jacobian_fn.expand()
        jacobian_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/jacobian_{frame_name}_{urdf_name}.casadi")

        test_lin_jac = ca.jacobian(clinear_velocity, q_dot_sym)
        lin_jac_fn = Function("lin_jac", [q_sym], [test_lin_jac])
        lin_jac_fn.expand()
        # lin_jac_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/lin_jac_{frame_name}.casadi")

        test_ang_jac = ca.jacobian(cangular_velocity, q_dot_sym)  # once taking derivate w.r.t q_dot_sym, the jacobian is only function of q_sym
        ang_jac_fn = Function("ang_jac", [q_sym], [test_ang_jac])
        ang_jac_fn.expand()
        # ang_jac_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/ang_jac_{frame_name}.casadi")


def load_forward_kinematics(urdf_name, urdf_path, frame_names):
    """
    Loads the saved forward kinematics casadi functions and runs them.
    """
    print("\n==== Load Casadi Functions ====")
    # Build the Pinocchio model from the URDF file
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    print("model name: " + model.name)
    q_num = pin.neutral(model)
    q_dot_num = np.random.rand(model.nv)
    print(f"q_num: {q_num.T}")
    print(f"q_dot_num: {q_dot_num.T}")
    Key_point_joint_idx = [model.getJointId(name) for name in key_point.Key_point_joint_names]
    print("Key_point_joint_idx:", Key_point_joint_idx)
    key_point_pos_fn_array = []
    key_point_rot_fn_array = []
    key_point_lin_vel_fn_array = []
    key_point_ang_vel_fn_array = []
    # 计算关键点坐标
    for i, joint_id in enumerate(Key_point_joint_idx):
        key_point_pos_bf_fn = Function.load(
            f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{key_point.Key_point_joint_names[i]}_pos_bf_{urdf_name}.casadi"
        )
        key_point_rot_bf_fn = Function.load(
            f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{key_point.Key_point_joint_names[i]}_rot_bf_{urdf_name}.casadi"
        )
        key_point_lin_vel_bf_fn = Function.load(
            f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{key_point.Key_point_joint_names[i]}_lin_vel_{urdf_name}.casadi"
        )
        key_point_ang_vel_bf_fn = Function.load(
            f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/key_point_{key_point.Key_point_joint_names[i]}_ang_vel_{urdf_name}.casadi"
        )
        key_point_pos_fn_array.append(key_point_pos_bf_fn(q_num))
        key_point_rot_fn_array.append(key_point_rot_bf_fn(q_num))
        key_point_lin_vel_fn_array.append(key_point_lin_vel_bf_fn(q_num, q_dot_num))
        key_point_ang_vel_fn_array.append(key_point_ang_vel_bf_fn(q_num, q_dot_num))
        print(f"Key point {key_point.Key_point_joint_names[i]} position in base frame:")
        print(np.array(key_point_pos_fn_array[i]))
        print(f"Key point {key_point.Key_point_joint_names[i]} rotation in base frame:")
        print(np.array(key_point_rot_fn_array[i]))
        print(f"Key point {key_point.Key_point_joint_names[i]} linear velocity:")
        print(np.array(key_point_lin_vel_fn_array[i]))
        print(f"Key point {key_point.Key_point_joint_names[i]} angular velocity:")
        print(np.array(key_point_ang_vel_fn_array[i]))
    # 动力学计算
    M_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/M_{urdf_name}.casadi")
    C_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/C_{urdf_name}.casadi")
    G_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/G_{urdf_name}.casadi")
    M = M_fn(q_num)
    C = C_fn(q_num, q_dot_num)
    G = G_fn(q_num)
    print(f"M(q): \n{np.array(M)}")
    print(f"C(q, q_dot): \n{np.array(C)}")
    print(f"G(q): \n{np.array(G)}")

    for frame_name in frame_names:
        fk_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/fk_{frame_name}_{urdf_name}.casadi")
        pos, rot = fk_fn(q_num)
        print(f"Frame {frame_name} \nposition: {np.array(pos.T)} \nrotation: \n{np.array(rot)}")

        jacobian_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/jacobian_{frame_name}_{urdf_name}.casadi")
        jacobian = jacobian_fn(q_num)
        print(f"Jacobian {frame_name}: \n{np.array(jacobian)}")

        # lin_jac_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/lin_jac_{frame_name}.casadi")
        # lin_jac = lin_jac_fn(q_num)
        # print(f"Lin Jaco {frame_name}: \n{np.array(lin_jac)}")

        # ang_jac_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/ang_jac_{frame_name}.casadi")
        # ang_jac = ang_jac_fn(q_num)
        # print(f"Ang Jaco {frame_name}: \n{np.array(ang_jac)}")

        # fk_right_foot_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/fk_right_foot.casadi")
        # fk_left_foot_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/fk_left_foot.casadi")
        # pos, rot = fk_right_foot_fn(q_num)
        # print(f"Frame right_foot \nposition: {np.array(pos.T)} \nrotation: \n{np.array(rot)}")
        # pos, rot = fk_left_foot_fn(q_num)
        # print(f"Frame left_foot \nposition: {np.array(pos.T)} \nrotation: \n{np.array(rot)}")

        # jacobian_right_foot_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/jacobian_right_foot.casadi")
        # jacobian_left_foot_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/jacobian_left_foot.casadi")

        # jacobian_right_foot = jacobian_right_foot_fn(q_num)
        # print(f"Jacobian right_foot: \n{np.array(jacobian_right_foot)}")
        # jacobian_left_foot = jacobian_left_foot_fn(q_num)
        # print(f"Jacobian left_foot: \n{np.array(jacobian_left_foot)}")


def _set_cuda_device(device_id: int) -> None:
    libcudart = ctypes.CDLL("libcudart.so")
    err = libcudart.cudaSetDevice(device_id)
    if err != 0:
        raise RuntimeError(f"cudaSetDevice({device_id}) returned error code {err}")


def dynamic_main_no_compile(cuda_device_id: int = 0):
    _set_cuda_device(cuda_device_id)
    print("==== casadi generator Start ====")
    urdf_name = "g1_29dof"
    urdf_path = f"{LABDYNAMIC_ROOT_DIR}/model/unitree_description/urdf/g1/main.urdf"
    frame_to_query = "right_ankle_roll_link"
    test_forward_kinematics_from_urdf(urdf_path, frame_to_query)
    frames_to_query = ["right_ankle_roll_link", "left_ankle_roll_link", "right_wrist_yaw_link", "left_wrist_yaw_link"]
    save_forward_kinematics(urdf_name, urdf_path, frames_to_query)
    load_forward_kinematics(urdf_name, urdf_path, frames_to_query)
    print("==== casadi generator Done ====")


def dynamic_main(cuda_device_id: int = 0):
    dynamic_main_no_compile(cuda_device_id)
    _set_cuda_device(cuda_device_id)
    print("==== ready to generator cusadi library ====")
    time.sleep(2)
    parser = cu.setupParser()
    args = parser.parse_args([])
    args.func_dir = PINOCCHIO_CASADI_FUNCTIONS_DIR
    args.clean_compile = True
    cu.printParserArguments(parser, args)
    cu.main(args)
    pass


if __name__ == "__main__":
    dynamic_main()
    pass

"""
# * Pinocchio tutorials
1. model = pin.buildModelFromUrdf(urdf_path) : Pinocchio Model object
2. data = model.createData() : Pinocchio Data object
    - pin.forwardKinematics(model, data, q) : Pinocchio uses model for the structure and writes the result into data
    - oMf: SE3 object describing the transform from the world frame to the frame
    - model.names.tolist()
    ['universe', 'a01_right_hip_yaw', 'a02_right_hip_abad', 'a03_right_hip_pitch', 'a04_right_knee', 'a05_right_ankle', 'a06_left_hip_yaw', 'a07_left_hip_abad', 'a08_left_hip_pitch', 'a09_left_knee', 'a10_left_ankle']

    Number of Jacobian = Number of Links
    
    = Joint order =
    Pinocchio: [x, y, z, qx, qy, qz, qw, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10]
    IsaacLab: [x, y, z, qw, qx, qy, qz, a01, a03, a05, a07, a09, a02, a04, a06, a08, a10]
    pin_to_isaaclab = [0, 1, 2, 6, 3, 4, 5, 7, 9, 11, 13, 15, 8, 10, 12, 14, 16]
    isaaclab_to_pin = [0, 1, 2, 4, 5, 6, 3, 7, 12, 8, 13, 9, 14, 10, 15, 11, 16]
 
3. # Compute separate Jacobians for linear and angular velocities
    test_lin_jac = ca.jacobian(clinear_velocity, q_dot_sym)
    lin_jac = Function("lin_jac", [q_sym], [test_lin_jac])
    lin_jac.expand()
    lin_jac.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/lin_jac_{frame_name}.casadi")

    test_ang_jac = ca.jacobian(cangular_velocity, q_dot_sym) # once taking derivate w.r.t q_dot_sym, the jacobian is only function of q_sym
    ang_jac = Function("ang_jac", [q_sym], [test_ang_jac])
    ang_jac.expand()
    ang_jac.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/ang_jac_{frame_name}.casadi")

    These each give a 3x16 jacobian matrix that maps joint positiono to linear or angular velocity of the frame.
    This perfectly matches with below:

    # Compute Jacobian
    jacobian = cpin.computeFrameJacobian(cmodel, cdata, q_sym, frame_id)
    jacobian_fn = Function("jacobian", [q_sym], [jacobian])
    jacobian_fn.expand()
    jacobian_fn.save(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/jacobian_{frame_name}.casadi")

4. The full body dynamics equation of motion : M(q) @ q_ddot + C(q, q_dot) @ q_dot + G(q) = tau + J_c.T @ F_c
    M = cpin.crba(cmodel, cdata, q_sym)
    C = cpin.computeCoriolisMatrix(cmodel, cdata, q_sym, q_dot_sym) @ q_dot_sym
    G = cpin.computeGeneralizedGravity(cmodel, cdata, q_sym)


# TODO
- Before saving the casadi fn, re-order the sequence to match the IsaacLab

"""
