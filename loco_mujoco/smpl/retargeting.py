import glob
from dataclasses import replace
import hashlib
import logging
import os
import time
from pathlib import Path
import json

import yaml
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

try:
    import joblib
    import torch
    from torch.autograd import Variable

    from loco_mujoco.smpl import SMPLH_Parser

    _OPTIONAL_IMPORT_INSTALLED = True
except ImportError as e:
    _OPTIONAL_IMPORT_INSTALLED = False
    _OPTIONAL_IMPORT_EXCEPTION = e

import loco_mujoco
from loco_mujoco import PATH_TO_SMPL_ROBOT_CONF
from loco_mujoco.core.mujoco_base import Mujoco
from loco_mujoco.core.utils.math import quat_scalarlast2scalarfirst
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from loco_mujoco.datasets.data_generation.utils import add_mocap_bodies
from loco_mujoco.smpl import SMPLH_BONE_ORDER_NAMES
from loco_mujoco.smpl.utils.smoothing import gaussian_filter_1d_batch
from loco_mujoco.environments import LocoEnv
from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
    interpolate_trajectories,
)
from loco_mujoco.datasets.data_generation import ExtendTrajData
from loco_mujoco.utils import setup_logger

OPTIMIZED_SHAPE_FILE_NAME = "shape_optimized.pkl"


def _fps_after_frame_skip(source_fps, skip: int) -> float:
    if skip <= 0:
        raise ValueError(f"skip must be positive, got {skip}")
    return float(np.asarray(source_fps).squeeze()) / float(skip)


def _compute_qvel_from_qpos(qpos: np.ndarray, fps: float, free_joint_name: str, model: mujoco.MjModel) -> np.ndarray:
    """
    Compute joint velocities from positions using MuJoCo's mj_differentiatePos.

    This uses MuJoCo's native differentiation which correctly handles quaternion
    differences for free joints, avoiding rotation vector singularities at π.

    Args:
        qpos: Joint positions array (T, nq)
        fps: Frequency of the trajectory in Hz
        free_joint_name: Name of the free joint (unused, kept for API compatibility)
        model: MuJoCo model

    Returns:
        Tuple of (qpos_trimmed, qvel) where first and last frames are removed due to central differencing
    """
    n_frames = len(qpos)
    dt = 1.0 / fps

    # Compute forward differences using MuJoCo's mj_differentiatePos
    qvel_fwd = np.zeros((n_frames - 1, model.nv), dtype=np.float64)
    for t in range(n_frames - 1):
        mujoco.mj_differentiatePos(model, qvel_fwd[t], dt, qpos[t], qpos[t + 1])

    # Central difference: average forward and backward velocities
    # qvel[t] = 0.5 * (v_fwd[t] + v_fwd[t-1]) for interior frames
    qvel = 0.5 * (qvel_fwd[1:] + qvel_fwd[:-1])

    # Trim qpos to match (remove first and last frames)
    qpos_trimmed = qpos[1:-1].copy()

    return qpos_trimmed, qvel


def _resolve_site_ids(model: mujoco.MjModel, site_names: list[str], *, context: str) -> list[int]:
    """Resolve MuJoCo site names and reject missing sites before indexing arrays."""
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name) for site_name in site_names]
    missing = [site_name for site_name, site_id in zip(site_names, site_ids, strict=True) if site_id < 0]
    if missing:
        raise ValueError(f"{context} references missing MuJoCo site(s): {', '.join(missing)}")
    return site_ids


def _resolve_smpl_joint_indices(sites_for_mimic: list[str], robot_conf: DictConfig, *, context: str) -> list[int]:
    """Map mimic sites to SMPL-H joint indices with clear configuration errors."""
    matches = robot_conf.site_joint_matches
    missing = [site_name for site_name in sites_for_mimic if site_name not in matches]
    if missing:
        raise ValueError(f"{context} is missing site_joint_matches entries for: {', '.join(missing)}")

    joint_indices = []
    invalid = []
    for site_name in sites_for_mimic:
        smpl_joint = matches[site_name].smpl_joint
        try:
            joint_indices.append(SMPLH_BONE_ORDER_NAMES.index(smpl_joint))
        except ValueError:
            invalid.append(f"{site_name}->{smpl_joint}")
    if invalid:
        raise ValueError(f"{context} references unknown SMPL-H joint(s): {', '.join(invalid)}")
    return joint_indices


def ensure_pytorch_precision_consistency():
    """
    Ensure consistent PyTorch precision across RTX and A100 GPUs.

    This function disables TensorFloat-32 (TF32) which is enabled by default on A100 GPUs
    but not available on RTX GPUs, causing precision differences in retargeting results.
    """
    try:
        import torch

        # Disable TensorFloat-32 for full FP32 precision consistency across GPU types
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Enable deterministic algorithms for reproducible results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set manual seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    except ImportError:
        # Torch not available, skip precision settings
        print("[Warning] PyTorch not installed, skipping precision consistency settings.")


def check_optional_imports():
    if not _OPTIONAL_IMPORT_INSTALLED:
        raise ImportError(
            f"[LocoMuJoCo] Optional smpl dependencies not installed. "
            f"Checkout the README for installation instructions. {_OPTIONAL_IMPORT_EXCEPTION}"
        )


def get_amass_dataset_path():
    path_to_conf = loco_mujoco.PATH_TO_VARIABLES
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_amass_datasets = data["LOCOMUJOCO_AMASS_PATH"]

    assert path_to_amass_datasets, "Please set the environment variable LOCOMUJOCO_AMASS_PATH."

    return path_to_amass_datasets


def get_converted_amass_dataset_path():
    path_to_conf = loco_mujoco.PATH_TO_VARIABLES
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_converted_amass_datasets = data["LOCOMUJOCO_CONVERTED_AMASS_PATH"]

    assert path_to_converted_amass_datasets, (
        "Please set the environment variable LOCOMUJOCO_CONVERTED_AMASS_PATH."
    )

    return path_to_converted_amass_datasets


def get_smpl_model_path():
    path_to_conf = loco_mujoco.PATH_TO_VARIABLES
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        path_to_smpl_model = data["LOCOMUJOCO_SMPL_MODEL_PATH"]

    assert path_to_smpl_model, "Please set the environment variable LOCOMUJOCO_SMPL_MODEL_PATH."

    return path_to_smpl_model


def load_amass_data(data_path: str) -> dict:
    """Load AMASS data from a file.

    Args:
        data_path (str): Path to the AMASS data file.


    Returns:
        dict: Parsed AMASS data including poses, translations, and other attributes.

    """
    path_to_amass_datasets = get_amass_dataset_path().replace("\\", "/")

    # get paths to all amass files
    path_to_all_amass_files = os.path.join(path_to_amass_datasets, "**/*.npz")
    all_pkls = glob.glob(path_to_all_amass_files, recursive=True)

    # get full dataset path
    key_names = [
        "/".join(data_path.replace("\\", "/").replace(path_to_amass_datasets + "/", "").split("/")).replace(".npz", "")
        for data_path in all_pkls]
    if data_path.startswith("/"):
        data_path = data_path[1:]
    data_path = data_path.replace(".npz", "")
    data_path = all_pkls[key_names.index(data_path)]

    # load data
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if "mocap_framerate" in entry_data:
        framerate = entry_data["mocap_framerate"]
    elif "mocap_frame_rate" in entry_data:
        framerate = entry_data["mocap_frame_rate"]
    else:
        raise ValueError("Framerate not found in the data file.")

    root_trans = entry_data["trans"]
    pose_aa = np.concatenate([entry_data["poses"][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = entry_data["betas"]
    gender = entry_data["gender"]

    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


def load_robot_conf_file(env_name: str):
    """Load a robot configuration file."""
    if "Mjx" in env_name:
        conf_name = env_name.replace("Mjx", "")
    else:
        conf_name = env_name
    filename = f"{conf_name}.yaml"
    filepath = os.path.join(PATH_TO_SMPL_ROBOT_CONF, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML file '{filename}' not found in path: {PATH_TO_SMPL_ROBOT_CONF}")
    default_conf = OmegaConf.load(PATH_TO_SMPL_ROBOT_CONF / "defaults.yaml")
    robot_conf = OmegaConf.load(filepath)
    robot_conf = OmegaConf.merge(default_conf, robot_conf)
    return robot_conf


def to_t_pose(env, robot_conf):
    """
    Set the humanoid to a T-pose by modifying the Mujoco Data structure.

    Args:
        env: environment.
        robot_conf: robot configuration file including the joint positions for the T-pose.

    """
    data = env._data
    # apply init pose modifiers
    for modifier in robot_conf.robot_pose_modifier:
        name, val = next(iter(modifier.items()))
        if name != "root":
            # convert string to numpy value
            val = np.array(eval(val))
            qpos_id = mj_jntname2qposid(name, env._model)
            data.qpos[qpos_id] += val
        else:
            # convert string to numpy value
            val = sRot.from_euler("xyz", eval(val), degrees=False).as_quat()
            val = quat_scalarlast2scalarfirst(val)
            data.qpos[3:7] += val


def fit_smpl_motion(
    env_name: str,
    robot_conf: DictConfig,
    path_to_smpl_model: str,
    motion_data: str | dict,
    path_to_optimized_smpl_shape: str,
    logger: logging.Logger,
    skip_steps: bool = True,
    visualize: bool = False,
) -> tuple[Trajectory, dict]:
    """Fit SMPL motion data to a robot configuration.

    Args:
        env_name (str): Name of the environment.
        robot_conf (DictConfig): Configuration of the robot.
        path_to_smpl_model (str): Path to the SMPL model.
        motion_data (Dict): Dict containing the motion data to process.
        path_to_optimized_smpl_shape (str): Path to the optimized SMPL shape file for the robot.
        logger (logging.Logger): Logger for status updates.
        visualize (bool): Whether to visualize the optimization process.

    Returns:
        tuple[Trajectory, dict]: The fitted motion trajectory and analysis data
            containing pos_error, retarget_fps, and site_names.

    """

    def get_xpos_and_xquat(smpl_positions, smpl_rot_mats, s2m_pos, s2m_rot_mat):
        # get rotations of mimic sites
        new_smpl_rot_mats = np.einsum("bij,bjk->bik", smpl_rot_mats, s2m_rot_mat)
        new_smpl_quat = sRot.from_matrix(new_smpl_rot_mats).as_quat()
        new_smpl_quat = quat_scalarlast2scalarfirst(new_smpl_quat)
        pos_offset = np.einsum("bij,bj->bi", new_smpl_rot_mats, s2m_pos)
        new_smpl_pos = torch.squeeze(smpl_positions - pos_offset)

        return new_smpl_pos, new_smpl_quat

    check_optional_imports()

    # Normalize environment name for consistent retargeting
    if "Mjx" in env_name:
        env_name = env_name.replace("Mjx", "")

    # get environment
    env_cls = Mujoco.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})

    # add mocap bodies for all 'site_for_mimic' instances of an environment
    mjspec = env.mjspec
    sites_for_mimic = env.sites_for_mimic
    _resolve_site_ids(env._model, sites_for_mimic, context=f"{env_name} sites_for_mimic")
    smpl2mimic_site_idx = _resolve_smpl_joint_indices(
        sites_for_mimic,
        robot_conf,
        context=f"{env_name} retargeting config",
    )
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mjspec = add_mocap_bodies(mjspec, sites_for_mimic, target_mocap_bodies, robot_conf, add_equality_constraint=True)
    env.reload_mujoco(mjspec)
    key = jax.random.key(0)
    env.reset(key)

    smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral")

    shape_new, scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale = joblib.load(
        path_to_optimized_smpl_shape
    )

    skip = robot_conf.optimization_params.skip_frames if skip_steps else 1
    pose_aa = torch.from_numpy(motion_data["pose_aa"][::skip]).float()
    pose_aa = torch.cat([pose_aa, torch.zeros((pose_aa.shape[0], 156 - pose_aa.shape[1]))], axis=-1)
    len_traj = pose_aa.shape[0]

    total_z_offset = offset_z + robot_conf.optimization_params.z_offset_feet
    trans = torch.from_numpy(motion_data["trans"][::skip]) + torch.tensor([0.0, 0.0, total_z_offset])

    # apply height scaling while preserving init height
    trans[:, :2] *= height_scale  # scale x and y
    trans[:, 2] = (trans[:, 2] - trans[0, 2]) * height_scale + trans[0, 2]

    with torch.no_grad():
        transformations_matrices = smpl_parser_n.get_joint_transformations(
            pose_aa.reshape(len_traj, -1, 3), shape_new.repeat(len_traj, 1), trans
        )
        global_pos = transformations_matrices[..., :3, 3]
        global_rot_mats = transformations_matrices[..., :3, :3].detach().numpy()
        root_pos = global_pos[:, 0:1]
        global_pos = (global_pos - global_pos[:, 0:1]) * scale.detach() + root_pos

    # calculate initial qpos from initial mocap pos
    init_mocap_pos, init_mocap_quat = get_xpos_and_xquat(
        global_pos[0, smpl2mimic_site_idx], global_rot_mats[0, smpl2mimic_site_idx], smpl2robot_pos, smpl2robot_rot_mat
    )
    qpos_init = get_init_qpos_for_motion_retargeting(env, init_mocap_pos, init_mocap_quat, robot_conf)
    env._data.qpos = qpos_init

    site_ids = _resolve_site_ids(env._model, sites_for_mimic, context=f"{env_name} retarget sites")
    qpos = np.zeros((len_traj, env._model.nq))
    num_matched = len(smpl2mimic_site_idx)
    site_xpos = np.zeros((len_traj, len(site_ids), 3), dtype=np.float32)
    site_xmat = np.zeros((len_traj, len(site_ids), 9), dtype=np.float32)

    dist_array = np.zeros((len_traj, num_matched))

    # max_pen = 0
    # TODO: currently with per frame offset, test will overall offset if needed for ground penetrations.
    t_start = time.perf_counter()
    for i in tqdm(range(len_traj)):
        mocap_pos, mocap_quat = get_xpos_and_xquat(
            global_pos[i, smpl2mimic_site_idx],
            global_rot_mats[i, smpl2mimic_site_idx],
            smpl2robot_pos,
            smpl2robot_rot_mat,
        )
        env._data.mocap_pos = mocap_pos
        env._data.mocap_quat = mocap_quat

        mujoco.mj_step(env._model, env._data, robot_conf.optimization_params.motion_iterations)

        qpos[i] = env._data.qpos.copy()
        mujoco.mj_forward(env._model, env._data)
        pen, _geom_id, _floor_id = max_penetration_with_floor(env._model, env._data)
        if pen < 0:
            qpos[i][2] -= pen
            env._data.qpos = qpos[i]
            mujoco.mj_forward(env._model, env._data)

        # TODO: test with overall ground penetration
        # max_pen = min(max_pen, pen)

        for k, site_id in enumerate(site_ids):
            current_pos = env._data.site_xpos[site_id]  # real robot site position
            target_pos = mocap_pos[k].cpu().numpy()  # target from SMPL
            dist_array[i, k] = np.linalg.norm(current_pos - target_pos)
            site_xpos[i, k] = current_pos
            site_xmat[i, k] = env._data.site_xmat[site_id]

        if visualize:
            env.render()

    t_total = time.perf_counter() - t_start
    retarget_fps = float(len_traj / max(1e-12, t_total))

    logger.info(f"[OK] SMPL retargeting complete: {len_traj} frames in {t_total:.2f}s ({retarget_fps:.2f} FPS)")

    # TODO: test later for overall ground penetrations
    # if max_pen < 0:
    #    qpos[:, 2] -= max_pen

    fps = _fps_after_frame_skip(motion_data["fps"], skip)

    # Compute qvel from qpos using helper function
    qpos, qvel = _compute_qvel_from_qpos(qpos, fps, env.root_free_joint_xml_name, env._model)
    site_xpos = site_xpos[1:-1]
    site_xmat = site_xmat[1:-1]

    njnt = env._model.njnt
    jnt_type = env._model.jnt_type
    jnt_names = [mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]

    site_ids_array = jnp.array(site_ids)
    traj_model = TrajectoryModel(
        njnt=njnt,
        jnt_type=jnp.array(jnt_type),
        nbody=env._model.nbody,
        body_rootid=jnp.array(env._model.body_rootid),
        body_weldid=jnp.array(env._model.body_weldid),
        body_mocapid=jnp.array(env._model.body_mocapid),
        body_pos=jnp.array(env._model.body_pos),
        body_quat=jnp.array(env._model.body_quat),
        body_ipos=jnp.array(env._model.body_ipos),
        body_iquat=jnp.array(env._model.body_iquat),
        nsite=len(site_ids),
        site_bodyid=jnp.array(env._model.site_bodyid)[site_ids_array],
        site_pos=jnp.array(env._model.site_pos)[site_ids_array],
        site_quat=jnp.array(env._model.site_quat)[site_ids_array],
    )
    traj_info = TrajectoryInfo(
        jnt_names,
        model=traj_model,
        frequency=fps,
        site_names=sites_for_mimic,
    )

    traj_data = TrajectoryData(
        jnp.array(qpos),
        jnp.array(qvel),
        site_xpos=jnp.array(site_xpos),
        site_xmat=jnp.array(site_xmat),
        split_points=jnp.array([0, len(qpos)]),
    )

    analysis = {
        "pos_error": dist_array,
        "retarget_fps": retarget_fps,
        "site_names": sites_for_mimic,
    }

    return Trajectory(traj_info, traj_data), analysis


def save_stage(path, model, qpos, **extra):
    """Save a small, named-coordinate diagnostic snapshot (not a Trajectory)."""
    mj = _mj()
    q = _array(qpos)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != model.nq:
        raise ValueError("qpos does not match model.nq")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, qpos=q,
        joint_names=np.asarray(_names(model, mj.mjtObj.mjOBJ_JOINT, model.njnt)),
        jnt_type=np.asarray(model.jnt_type),
        jnt_qposadr=np.asarray(model.jnt_qposadr),
        jnt_limited=np.asarray(model.jnt_limited),
        jnt_range=np.asarray(model.jnt_range),
        mujoco_version=np.asarray(mj.__version__), **extra,
    )


def hinge_report(model, qpos):
    """Report scalar hinge ranges and frame-to-frame jumps, in degrees."""
    mj = _mj()
    q = _array(qpos)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != model.nq:
        raise ValueError("qpos does not match model.nq")
    result = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
            continue
        a = int(model.jnt_qposadr[j])
        v = q[:, a]
        lo, hi = model.jnt_range[j]
        violation = np.maximum(np.maximum(lo-v, v-hi), 0)
        result.append({
            "joint": model.joint(j).name,
            "min_deg": float(np.rad2deg(v.min())),
            "max_deg": float(np.rad2deg(v.max())),
            "limited": bool(model.jnt_limited[j]),
            "range_deg": np.rad2deg([lo, hi]).tolist(),
            "max_violation_deg": (float(np.rad2deg(violation.max()))
                                   if model.jnt_limited[j] else None),
            "max_step_deg": (float(np.rad2deg(np.abs(np.diff(v)).max()))
                             if len(v) > 1 else 0.0),
        })
    return result

def bounded_initial_qpos(env, init_mocap_pos, init_mocap_quat, robot_conf):
    """Optional replacement for get_init_qpos_for_motion_retargeting.

    A kinematic, bound-constrained first-frame fit, not another dynamics rollout.
    Supports ONE world-attached free root plus scalar hinge/slide joints.
    Uses a rest-pose preference to discourage gratuitous axial rotation and a
    second, bent-knee seed to reduce straight-leg initialization ambiguity.
    Respects WELD vs CONNECT site target types; soft-constraint parameters are
    NOT reproduced by this least-squares objective. No collision or muscle-
    feasibility guarantee. Later mj_step calls can still leave these bounds.

    Optional optimization_params keys:
      init_ik_max_nfev: 150
      init_ik_posture_weight: 0.02  (metre-equivalent residual per radian)
      init_ik_max_position_error: 0.08  (metres; explicit acceptance criterion)
      init_ik_max_orientation_error_deg: 90.0  (only weighted WELD targets)
      init_ik_joint_bounds: {joint_name: [lower_rad, upper_rad]}
    All values are engineering settings, not anatomical recommendations.
    """
    mj = _mj()
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    opt = robot_conf.optimization_params
    if bool(opt.get("disable_joint_limits", False)):
        raise ValueError("Set disable_joint_limits=False BEFORE constructing the fitting model")
    model = env._model
    free = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or np.any(model.jnt_type == mj.mjtJoint.mjJNT_BALL):
        raise ValueError("Initializer supports one free root and scalar joints only")
    root = int(free[0])
    body = int(model.jnt_bodyid[root])
    if int(model.body_parentid[body]) != 0:
        raise ValueError("Free root must be attached directly to world")
    ra = int(model.jnt_qposadr[root])
    sj = np.asarray([j for j in range(model.njnt) if j != root], dtype=int)
    sa = np.asarray(model.jnt_qposadr[sj], dtype=int)
    sites = list(env.sites_for_mimic)
    ids = [_id(model, mj.mjtObj.mjOBJ_SITE, s) for s in sites]
    pos = _array(init_mocap_pos).reshape(-1, 3)
    quat = _array(init_mocap_quat).reshape(-1, 4)
    if pos.shape != (len(ids), 3) or quat.shape != (len(ids), 4):
        raise ValueError("Target arrays must follow env.sites_for_mimic exactly")
    if not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError("Nonfinite mocap targets")
    target_R = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    rotation_weights = []
    for name in sites:
        conf = robot_conf.site_joint_matches[name]
        eq = str(conf.get("equality_constraint_type", "mjEQ_WELD"))
        if eq not in ("mjEQ_WELD", "mjEQ_CONNECT"):
            raise ValueError(f"Unsupported site constraint {eq}")
        rotation_weights.append(float(conf.get("torque_scale", 1.0))
                                if eq == "mjEQ_WELD" else 0.0)
    rotation_weights = np.asarray(rotation_weights)
    if np.any(rotation_weights < 0) or not np.isfinite(rotation_weights).all():
        raise ValueError("Invalid orientation weights")

    lo = np.full(len(sj), -np.inf)
    hi = np.full(len(sj), np.inf)
    for k, j in enumerate(sj):
        if model.jnt_limited[j]:
            lo[k], hi[k] = model.jnt_range[j]
    overrides = opt.get("init_ik_joint_bounds", {}) or {}
    for name, limits in overrides.items():
        j = _id(model, mj.mjtObj.mjOBJ_JOINT, name)
        matches = np.flatnonzero(sj == j)
        if len(matches) != 1:
            raise ValueError(f"Bounds override is not a scalar joint: {name}")
        k = int(matches[0])
        newlo, newhi = map(float, limits)
        if not np.isfinite([newlo, newhi]).all():
            raise ValueError(f"Nonfinite override: {name}")
        lo[k], hi[k] = max(lo[k], newlo), min(hi[k], newhi)
    if np.any(lo >= hi):
        raise ValueError("Empty/zero-width joint interval; use the model's locked joints instead")

    # Start from a valid rest pose, not from the corrupted legacy coordinates.
    qbase = np.asarray(model.qpos0, dtype=float).copy()
    qbase[sa] = np.clip(qbase[sa], lo, hi)  # A seed only, never the final repair.
    data = mj.MjData(model)
    data.qpos[:] = qbase
    mj.mj_kinematics(model, data)
    root_sites = [k for k, sid in enumerate(ids) if model.site_bodyid[sid] == body]
    if not root_sites:
        raise ValueError("Need a mimic site directly on the free-root body")
    k = root_sites[0]
    R0 = Rotation.from_quat(qbase[ra+3:ra+7][[1, 2, 3, 0]])
    current_site_R = Rotation.from_matrix(data.site_xmat[ids[k]].reshape(3, 3))
    delta_R = target_R[k] * current_site_R.inv()
    qbase[ra:ra+3] = pos[k] - delta_R.apply(data.site_xpos[ids[k]] - qbase[ra:ra+3])
    Rbase = delta_R * R0
    rest = qbase[sa].copy()
    posture_weight = float(opt.get("init_ik_posture_weight", 0.02))
    if not np.isfinite(posture_weight) or posture_weight < 0:
        raise ValueError("Invalid posture weight")
    pweights = np.full(len(sj), posture_weight)
    # Named preference, not wrong-index clipping; still a soft prior, not a ban.
    for k, j in enumerate(sj):
        if model.joint(int(j)).name in ("hip_rotation_l", "hip_rotation_r"):
            pweights[k] *= 3.0

    def unpack(x):
        q = qbase.copy()
        q[ra:ra+3] = x[:3]
        rq = (Rotation.from_rotvec(x[3:6]) * Rbase).as_quat()
        q[ra+3:ra+7] = rq[[3, 0, 1, 2]]
        q[sa] = x[6:]
        return q

    def errors(q):
        data.qpos[:] = q
        mj.mj_kinematics(model, data)
        dp = data.site_xpos[ids] - pos
        curr = Rotation.from_matrix(data.site_xmat[ids].reshape(-1, 3, 3))
        dr = (target_R.inv() * curr).as_rotvec()
        return dp, dr

    def residual(x):
        dp, dr = errors(unpack(x))
        return np.concatenate((dp.ravel(), (dr*rotation_weights[:, None]).ravel(),
                               (x[6:]-rest)*pweights))

    x0 = np.r_[qbase[ra:ra+3], np.zeros(3), rest]
    lower = np.r_[np.full(6, -np.inf), lo]
    upper = np.r_[np.full(6, np.inf), hi]
    seeds = [x0]
    bend = x0.copy()
    for k, j in enumerate(sj):
        if "knee" in (model.joint(int(j)).name or "") and np.isfinite([lo[k], hi[k]]).all():
            bend[6+k] = 0.5*(lo[k]+hi[k])
    if not np.array_equal(bend, x0):
        seeds.append(bend)
    candidates = []
    for seed in seeds:
        fit = least_squares(residual, seed, bounds=(lower, upper), method="trf",
                            x_scale="jac", max_nfev=int(opt.get("init_ik_max_nfev", 150)))
        q = unpack(fit.x)
        dp, dr = errors(q)
        candidates.append((fit.cost, fit, q, dp, dr))
    _, fit, q, dp, dr = min(candidates, key=lambda item: item[0])
    poserr = np.linalg.norm(dp, axis=1)
    angerr = np.rad2deg(np.linalg.norm(dr, axis=1))
    enabled = rotation_weights > 0
    report = {"optimizer_success": bool(fit.success), "message": str(fit.message),
              "max_position_error_m": float(poserr.max()),
              "max_weighted_site_orientation_error_deg":
                  float(angerr[enabled].max()) if enabled.any() else 0.0,
              "worst_position_site": sites[int(poserr.argmax())],
              "nfev_best_seed": int(fit.nfev),
              "hinges": hinge_report(model, q)}
    logging.getLogger(__name__).info("Bounded initializer: %s", json.dumps(report))
    if np.any(q[sa] < lo-1e-8) or np.any(q[sa] > hi+1e-8):
        raise AssertionError("Bounded optimizer returned an invalid scalar coordinate")
    # Caller assigns qpos. Do not leave spurious initial motion/warm-start values.
    env._data.qvel[:] = 0
    env._data.qacc_warmstart[:] = 0
    return q

def get_init_qpos_for_motion_retargeting(
    env,
    init_mocap_pos,
    init_mocap_quat,
    robot_conf,
):
    return bounded_initial_qpos(
        env,
        init_mocap_pos,
        init_mocap_quat,
        robot_conf,
    )


def fit_smpl_shape(
    env_name: str,
    robot_conf: DictConfig,
    path_to_smpl_model: str,
    save_path_new_smpl_shape: str,
    logger: logging.Logger,
    visualize: bool = False,
) -> None:
    """Fit the SMPL shape to match the robot configuration.

    Args:
        env_name (str): Name of the environment.
        robot_conf (DictConfig): Configuration of the robot.
        path_to_smpl_model (str): Path to the SMPL model.
        save_path_new_smpl_shape (str): Path to save the optimized shape.
        logger (logging.Logger): Logger for status updates.
        visualize (bool): Whether to visualize the optimization process.

    """

    check_optional_imports()

    # Ensure consistent PyTorch precision across different devices
    ensure_pytorch_precision_consistency()

    z_offset_viz = 2.0  # for visualization only

    # get environment
    env_cls = Mujoco.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})

    # add mocap bodies for all 'site_for_mimic' instances of an environment
    mjspec = env.mjspec
    sites_for_mimic = env.sites_for_mimic
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mjspec = add_mocap_bodies(mjspec, sites_for_mimic, target_mocap_bodies, robot_conf, add_equality_constraint=False)
    env.reload_mujoco(mjspec)
    key = jax.random.key(0)
    env.reset(key)

    smpl2mimic_site_idx = []
    for s in sites_for_mimic:
        # find smpl name
        for site_name, conf in robot_conf.site_joint_matches.items():
            if site_name == s:
                smpl2mimic_site_idx.append(SMPLH_BONE_ORDER_NAMES.index(conf.smpl_joint))

    # set humanoid to T-pose
    to_t_pose(env, robot_conf)

    # save initial qpos
    qpos_init = env._data.qpos.copy()
    qpos_init[0:3] = 0
    qpos_init[2] = z_offset_viz

    # set initial qpos and forward
    env._data.qpos = qpos_init
    mujoco.mj_forward(env._model, env._data)

    # get joint names
    device = torch.device(robot_conf.optimization_params.torch_device)

    # get initial pose
    pose_aa_stand = np.zeros((1, 156)).reshape(-1, 52, 3)
    pose_aa_stand[:, SMPLH_BONE_ORDER_NAMES.index("Pelvis")] = sRot.from_euler(
        "xyz", [np.pi / 2, 0.0, np.pi / 2], degrees=False
    ).as_rotvec()
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 156)).requires_grad_(False)

    # setup parser
    smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral")

    # define optimization variables
    shape_new = Variable(torch.zeros([1, 16]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    trans = torch.zeros([1, 3]).requires_grad_(False)
    optimizer_shape = torch.optim.Adam([shape_new, scale], lr=robot_conf.optimization_params.shape_lr)

    # get target site positions
    site_ids = [mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, s) for s in sites_for_mimic]
    target_site_pos = torch.from_numpy(env._data.site_xpos[site_ids])[None]
    target_site_mat = torch.from_numpy(env._data.site_xmat[site_ids])

    # get z offset
    z_offset = torch.tensor([0.0, 0.0, z_offset_viz])[None]

    # get the transformation matrices
    transformations_matrices = smpl_parser_n.get_joint_transformations(pose_aa_stand, shape_new, trans)
    global_rot_mats = transformations_matrices.detach().numpy()[..., :3, :3]
    global_pos = transformations_matrices.detach().numpy()[..., :3, 3]
    global_rot_mats = sRot.from_matrix(global_rot_mats[0, smpl2mimic_site_idx])

    # get rotations of mimic sites
    target_site_mat = sRot.from_matrix(target_site_mat.detach().numpy().reshape(-1, 3, 3))

    # rel rotation smpl to robot
    smpl2robot_rot_mat = np.einsum("bij,bjk->bik", global_rot_mats.inv().as_matrix(), target_site_mat.as_matrix())

    # transform smpl rotations to match robot site rotations
    # (here done just for visualization, only used in motion_fit function)
    new_smpl_rot_mats = np.einsum("bij,bjk->bik", global_rot_mats.as_matrix(), smpl2robot_rot_mat)
    new_smpl_quats = quat_scalarlast2scalarfirst(sRot.from_matrix(new_smpl_rot_mats).as_quat())

    pbar = tqdm(range(robot_conf.optimization_params.shape_iterations))
    init_feet_z_pos = None
    init_head_z_pos = None
    for iteration in pbar:
        transformations_matrices = smpl_parser_n.get_joint_transformations(pose_aa_stand, shape_new, trans)
        global_pos = transformations_matrices[..., :3, 3]

        if init_feet_z_pos is None:
            init_feet_z_pos = np.minimum(
                global_pos[0, SMPLH_BONE_ORDER_NAMES.index("R_Ankle"), 2].detach().numpy(),
                global_pos[0, SMPLH_BONE_ORDER_NAMES.index("L_Ankle"), 2].detach().numpy(),
            )
            init_head_z_pos = global_pos[0, SMPLH_BONE_ORDER_NAMES.index("Head"), 2].detach().numpy()

        if iteration == robot_conf.optimization_params.shape_iterations - 1:
            final_feet_z_pos = np.minimum(
                global_pos[0, SMPLH_BONE_ORDER_NAMES.index("R_Ankle"), 2].detach().numpy(),
                global_pos[0, SMPLH_BONE_ORDER_NAMES.index("L_Ankle"), 2].detach().numpy(),
            )
            final_head_z_pos = global_pos[0, SMPLH_BONE_ORDER_NAMES.index("Head"), 2].detach().numpy()

        global_pos = (global_pos - global_pos[:, 0] + z_offset) * scale

        if visualize:
            env._data.qpos = qpos_init
            env._data.mocap_pos = torch.squeeze(global_pos[:, smpl2mimic_site_idx]).detach().numpy()
            env._data.mocap_quat = new_smpl_quats

            mujoco.mj_forward(env._model, env._data)
            env.render()

        # calculate loss
        diff = target_site_pos - global_pos[:, smpl2mimic_site_idx]
        loss = diff.norm(dim=-1).mean()

        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_shape.step()

    # save positions offset
    smpl_pos = global_pos[0, smpl2mimic_site_idx].detach().numpy()
    smpl2robot_pos = smpl_pos - target_site_pos.detach().numpy()
    smpl2robot_pos = np.squeeze(smpl2robot_pos)

    # save new z-offset
    offset_z = init_feet_z_pos - final_feet_z_pos
    height_scale = (final_head_z_pos - final_feet_z_pos) / (init_head_z_pos - init_feet_z_pos)

    # Extract the directory path from the save path
    directory = os.path.dirname(save_path_new_smpl_shape)

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # save
    joblib.dump(
        (shape_new.detach(), scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale),
        save_path_new_smpl_shape,
    )
    logger.info(f"Shape parameters saved at {save_path_new_smpl_shape}")


def max_penetration_with_floor(model, data, floor_name="floor", verbose=False):
    """
    Returns:
        (deepest_penetration, (geom_id, floor_id))
    where:
        deepest_penetration < 0  → penetration depth
        = 0 → no penetration
    """

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_name)
    if floor_id < 0:
        raise ValueError(f"No geom named '{floor_name}' found")

    max_pen = 0.0
    offending_geom = None

    for i in range(data.ncon):
        c = data.contact[i]

        # Check if either side is FLOOR
        if c.geom1 == floor_id or c.geom2 == floor_id:
            if c.dist < max_pen:
                max_pen = c.dist
                offending_geom = c.geom1 if c.geom2 == floor_id else c.geom2

    if offending_geom is not None and verbose:
        geom_name = model.geom(offending_geom).name
        logging.debug(f"Penetration {max_pen:.6f} m between geom {geom_name} and FLOOR ({floor_id})")

    return max_pen, offending_geom, floor_id


def motion_transfer_robot_to_robot(
    env_name_source: str,
    robot_conf_source: DictConfig,
    traj_source: Trajectory,
    path_source_robot_smpl_data: str,
    env_name_target: str,
    robot_conf_target: DictConfig,
    path_target_robot_smpl_data: str,
    path_to_smpl_model: str,
    logger: logging.Logger,
    path_to_fitted_motion_source: str | None = None,
    visualize: bool = False,
) -> Trajectory:
    def rotation_matrix_loss_geodesic(rot1, rot2):
        """
        Computes the geodesic distance loss between two rotation matrices with two batch dimensions.

        rot1, rot2: (B1, B2, 3, 3)
        Returns: Mean geodesic loss over (B1, B2)
        """
        rot_diff = torch.matmul(rot1.transpose(-2, -1), rot2)  # rot1^T * rot2, supports (B1, B2, 3, 3)
        trace = torch.einsum("...ii->...", rot_diff)  # Extract trace along last two dims (B1, B2)
        eps = 1e-6  # Small epsilon for numerical stability
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0 + eps, 1.0 - eps))
        return theta.mean()

    check_optional_imports()

    path_to_target_robot_smpl_shape = os.path.join(path_target_robot_smpl_data, OPTIMIZED_SHAPE_FILE_NAME)

    if path_to_fitted_motion_source is not None and not os.path.exists(path_to_fitted_motion_source):
        device = torch.device("cuda")

        # get the source env
        env_cls = Mujoco.registered_envs[env_name_source]
        env = env_cls(**robot_conf_source.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})

        # add mocap bodies for all 'site_for_mimic' instances of an environment
        mjspec = env.mjspec
        sites_for_mimic = env.sites_for_mimic
        target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
        mjspec = add_mocap_bodies(
            mjspec, sites_for_mimic, target_mocap_bodies, robot_conf_source, add_equality_constraint=False
        )
        env.reload_mujoco(mjspec)
        key = jax.random.key(0)
        env.reset(key)

        # extend the trajectory to include more model-specific entities
        traj_source = extend_motion(env_name_source, robot_conf_source.env_params, traj_source, logger)

        # load the source trajectory
        env.load_trajectory(traj_source, warn=False)

        # convert traj to numpy
        env.th.to_numpy()

        # get the body_shape of the source robot
        path_to_source_robot_smpl_shape = os.path.join(path_source_robot_smpl_data, OPTIMIZED_SHAPE_FILE_NAME)
        if not os.path.exists(path_to_source_robot_smpl_shape):
            logger.info("Robot shape file not found, fitting new one ...")
            fit_smpl_shape(
                env_name_source, robot_conf_source, path_to_smpl_model, path_to_source_robot_smpl_shape, logger
            )
        else:
            logger.info(f"Found existing robot shape file at {path_to_source_robot_smpl_shape}")
        (
            shape_source,
            scale_source,
            smpl2robot_pos_source,
            smpl2robot_rot_mat_source,
            offset_z_source,
            height_scale_source,
        ) = joblib.load(path_to_source_robot_smpl_shape)

        # get the source site positions used as a target for optimization
        sites_for_mimic = env.sites_for_mimic
        site_ids = np.array([mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, s) for s in sites_for_mimic])
        target_site_pos = torch.from_numpy(env.th.traj.data.site_xpos[:, site_ids])
        target_site_mat = torch.from_numpy(env.th.traj.data.site_xmat[:, site_ids])
        len_dataset = env.th.traj.data.n_samples

        # define the optimization variables
        pose = np.zeros([len_dataset, 156]).reshape(-1, 52, 3)
        init_rot_mat = target_site_mat[:, sites_for_mimic.index("pelvis_mimic")].reshape(-1, 3, 3)
        init_rot_mat = np.einsum(
            "nij,jk->nik", init_rot_mat, np.linalg.inv(smpl2robot_rot_mat_source[sites_for_mimic.index("pelvis_mimic")])
        )
        pose[:, SMPLH_BONE_ORDER_NAMES.index("Pelvis")] = sRot.from_matrix(init_rot_mat).as_rotvec()
        pose = torch.from_numpy(pose.reshape(-1, 156))
        pose = Variable(pose.float().to(device), requires_grad=True)
        trans = target_site_pos[:, sites_for_mimic.index("pelvis_mimic")].clone().to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([pose, trans], lr=robot_conf_source.optimization_params.pose_lr)
        scale_source = torch.tensor(scale_source).float().to(device).detach()

        # setup parser
        smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral").to(device)

        smpl2mimic_site_idx = []
        for s in sites_for_mimic:
            # find smpl name
            for site_name, conf in robot_conf_source.site_joint_matches.items():
                if site_name == s:
                    smpl2mimic_site_idx.append(SMPLH_BONE_ORDER_NAMES.index(conf.smpl_joint))

        shape_source = shape_source.repeat(len_dataset, 1).detach().to(device)

        # convert target site poses from site frame to smpl frame
        robot2smpl_pos_source = -smpl2robot_pos_source
        robot2smpl_rot_mat_source = np.linalg.inv(smpl2robot_rot_mat_source)
        pos_offset = np.einsum("nbij,bj->nbi", target_site_mat.reshape(len_dataset, -1, 3, 3), robot2smpl_pos_source)
        target_site_mat = np.einsum(
            "nbij,bjk->nbik", target_site_mat.reshape(len_dataset, -1, 3, 3), robot2smpl_rot_mat_source
        )
        target_site_pos = target_site_pos - pos_offset

        # convert to torch
        target_site_pos = target_site_pos.float().to(device)
        target_site_mat = torch.from_numpy(target_site_mat).float().to(device)

        iterations = robot_conf_source.optimization_params.pose_iterations
        pos_loss_weight = robot_conf_source.optimization_params.pos_loss_weight
        rot_loss_weight = robot_conf_source.optimization_params.rot_loss_weight
        for _i in tqdm(range(iterations)):
            # sample random indices
            transformations_matrices = smpl_parser_n.get_joint_transformations(pose, shape_source, trans)

            # get the global positions and rotations
            global_pos = transformations_matrices[..., :3, 3]
            global_rot_mats = transformations_matrices[..., :3, :3]

            # scale
            global_pos = (global_pos - global_pos[:, 0:1]) * scale_source + global_pos[:, 0:1]

            # calculate the loss
            pos_loss = (target_site_pos - global_pos[:, smpl2mimic_site_idx]).norm(dim=-1).mean()
            mat_loss = rotation_matrix_loss_geodesic(
                global_rot_mats[:, smpl2mimic_site_idx], target_site_mat.reshape(len_dataset, -1, 3, 3)
            )
            root_consistency_loss = torch.norm(pose[:1, :3] - pose[1:, :3], p=2).mean()

            if torch.any(torch.isnan(mat_loss)):
                raise ValueError("NaN in rotation matrix loss.")

            loss = pos_loss_weight * pos_loss + rot_loss_weight * mat_loss + 0.0 * root_consistency_loss

            if visualize:
                index = 0

                # convert smpl frame to site frame for visualization
                new_global_pos = global_pos[index, smpl2mimic_site_idx].cpu().detach().numpy()
                new_smpl_rot_mats = np.einsum(
                    "bij,bjk->bik",
                    global_rot_mats[index, smpl2mimic_site_idx].cpu().detach().numpy(),
                    smpl2robot_rot_mat_source,
                )
                pos_offset = np.einsum("bij,bj->bi", new_smpl_rot_mats, smpl2robot_pos_source)
                new_global_pos = np.squeeze(new_global_pos - pos_offset)

                env._data.mocap_pos = new_global_pos
                env._data.mocap_quat = quat_scalarlast2scalarfirst(sRot.from_matrix(new_smpl_rot_mats).as_quat())
                env._data.qpos = env.th.traj.data.qpos[index]
                mujoco.mj_forward(env._model, env._data)
                env.render()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # apply smoothing
            kernel_size = robot_conf_target.optimization_params.smoothing_kernel_size
            sigma = robot_conf_target.optimization_params.smoothing_sigma
            if sigma > 0:
                pose_non_filtered = pose[:, 3:].reshape(len_dataset, -1, 3)
                pose_non_filtered = pose_non_filtered.permute(1, 2, 0)
                pose_filtered = gaussian_filter_1d_batch(pose_non_filtered, kernel_size, sigma, device)
                pose_filtered = pose_filtered.permute(2, 0, 1)
                pose.data[:, 3:] = pose_filtered.reshape(-1, 153)

        motion_file = {
            "pose_aa": pose.cpu().detach().numpy().reshape(-1, 156),
            "trans": trans.cpu().detach().numpy(),
            "fps": env.th.traj.info.frequency,
        }

        # account for scale
        motion_file["pose_aa"] /= scale_source.cpu().detach().numpy()
        motion_file["trans"][:, 2] -= offset_z_source

        # apply height scaling while preserving init height
        height_scale_source_inv = 1 / height_scale_source
        trans[:, :2].data *= height_scale_source_inv  # scale x and y
        trans[:, 2].data = (trans[:, 2] - trans[0, 2]) * height_scale_source_inv + trans[0, 2]

        if path_to_fitted_motion_source is not None:
            # create dir if it does not exist
            directory = os.path.dirname(path_to_fitted_motion_source)
            os.makedirs(directory, exist_ok=True)
            # if a file path is provided, save the fitted motion
            np.savez(path_to_fitted_motion_source, **motion_file)

    else:
        logger.info(f"Loading fitted motion from {path_to_fitted_motion_source}.")
        motion_file = np.load(path_to_fitted_motion_source)

    # generate the body_shape of the target robot if it does not exist
    if not os.path.exists(path_to_target_robot_smpl_shape):
        logger.info("Robot shape file not found, fitting new one ...")
        fit_smpl_shape(
            env_name_target, robot_conf_target, path_to_smpl_model, path_to_target_robot_smpl_shape, logger, visualize
        )
    else:
        logger.info(f"Found existing robot shape file at {path_to_target_robot_smpl_shape}")

    traj_target, _ = fit_smpl_motion(
        env_name_target,
        robot_conf_target,
        path_to_smpl_model,
        motion_file,
        path_to_target_robot_smpl_shape,
        logger,
        skip_steps=False,
        visualize=visualize,
    )

    return traj_target


def validate_reference_data(traj: Trajectory) -> None:
    """Validate derived-array dimensions against the trajectory's own metadata."""
    data, model = traj.data, traj.info.model
    n = int(data.qpos.shape[0])
    if n == 0:
        raise ValueError("The reference trajectory has no samples.")
    expected = {
        "xpos": (n, int(model.nbody), 3),
        "xquat": (n, int(model.nbody), 4),
        "cvel": (n, int(model.nbody), 6),
        "subtree_com": (n, int(model.nbody), 3),
        "site_xpos": (n, int(model.nsite), 3),
        "site_xmat": (n, int(model.nsite), 9),
    }
    errors = []
    for name, shape in expected.items():
        value = getattr(data, name, None)
        if value is None or tuple(value.shape) != shape:
            errors.append(f"{name}: expected {shape}, got {getattr(value, 'shape', None)}")
        elif not np.isfinite(np.asarray(value)).all():
            errors.append(f"{name}: contains non-finite values")
    for name in ("qpos", "qvel"):
        value = np.asarray(getattr(data, name))
        if value.ndim != 2 or value.shape[0] != n or value.shape[1] == 0:
            errors.append(f"{name}: invalid shape {value.shape}")
        elif not np.isfinite(value).all():
            errors.append(f"{name}: contains non-finite values")
    if errors:
        raise ValueError("Incomplete or invalid reference data:\n" + "\n".join(errors))

def box_clearances(model, data, floor, gids):
    """Exact lowest-point height for oriented boxes over a horizontal plane."""
    F = np.asarray(data.geom_xmat[floor]).reshape(3, 3)
    if not np.allclose(F[:, 2], [0, 0, 1], atol=1e-8):
        raise ValueError("Floor is not horizontal and upward-facing")
    R = np.asarray(data.geom_xmat[gids]).reshape(-1, 3, 3)
    return (data.geom_xpos[gids, 2] - data.geom_xpos[floor, 2]
            - np.sum(np.abs(R[:, 2, :])*model.geom_size[gids], axis=1))


def _array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _mj():
    import mujoco
    return mujoco


def _id(model, kind, name):
    value = _mj().mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"Missing MuJoCo object {name!r}")
    return int(value)


def _names(model, kind, count):
    mj = _mj()
    return [mj.mj_id2name(model, kind, i) for i in range(count)]


def _selected_box_geoms(model, names, root_joint_name, floor_name):
    mj = _mj()
    root = _id(model, mj.mjtObj.mjOBJ_JOINT, root_joint_name)
    if model.jnt_type[root] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError("Selected root is not a free joint")
    root_body = int(model.jnt_bodyid[root])
    if int(model.body_parentid[root_body]) != 0:
        raise ValueError("Selected root must be world-attached")
    floor = _id(model, mj.mjtObj.mjOBJ_GEOM, floor_name)
    fb = int(model.geom_bodyid[floor])
    if model.geom_type[floor] != mj.mjtGeom.mjGEOM_PLANE or model.body_weldid[fb] != 0:
        raise ValueError("Grounding requires a static horizontal plane")
    if isinstance(names, str) or not names:
        raise ValueError("Specify the actual collision BOX geom names, as a list")
    gids = [_id(model, mj.mjtObj.mjOBJ_GEOM, name) for name in names]
    pairs = {tuple(sorted((int(a), int(b))))
             for a, b in zip(model.pair_geom1, model.pair_geom2)}
    for name, gid in zip(names, gids):
        if model.geom_type[gid] != mj.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"{name} is not a box: this narrow helper does not guess mesh support geometry")
        if int(model.body_rootid[model.geom_bodyid[gid]]) != root_body:
            raise ValueError(f"{name} is not attached to the selected root")
        explicit = tuple(sorted((gid, floor))) in pairs
        mask = bool((int(model.geom_contype[gid]) & int(model.geom_conaffinity[floor]))
                    or (int(model.geom_contype[floor]) & int(model.geom_conaffinity[gid])))
        if not (explicit or mask):
            raise ValueError(f"{name} is not floor-collision eligible by mask or explicit pair")
    return root, floor, gids

def ground_final_reference(traj, model, *, root_joint_name="root",
                           foot_geom_names=("foot_box_l", "foot_box_r"),
                           floor_name="floor", mode="global", clearance=0.0,
                           max_shift=0.25):
    """Ground an already-good, final-rate, fully model-aligned reference.

    GLOBAL: one root-Z shift per clip; preserves genuine vertical motion.
    PER_FRAME: each frame's lowest box touches; ONLY continuous foot support.
    Does NOT repair twisting, guarantee balance, or infer contact/flight.
    Non-root-Z qpos coordinates are checked for EXACT equality before return.
    Recomputes all qvel using MuJoCo differences (per clip) and body/site caches.
    Does not advance physics. Returned arrays retain the input NumPy/JAX backend.
    Does not save files. Do not subsequently rescale/resample without rechecking.
    """
    mj = _mj()
    if mode not in ("global", "per_frame"):
        raise ValueError("mode must be global or per_frame")
    if not np.isfinite([clearance, max_shift]).all() or clearance < 0 or max_shift <= 0:
        raise ValueError("Invalid clearance/max_shift")
    if getattr(traj, "transitions", None) is not None:
        raise ValueError("Do not modify a trajectory with already-built policy transitions")
    joint_names = _names(model, mj.mjtObj.mjOBJ_JOINT, model.njnt)
    body_names = _names(model, mj.mjtObj.mjOBJ_BODY, model.nbody)
    site_names = _names(model, mj.mjtObj.mjOBJ_SITE, model.nsite)
    for field, expected in (("joint_names", joint_names), ("body_names", body_names),
                            ("site_names", site_names)):
        got = getattr(traj.info, field, None)
        if got is None or list(got) != expected:
            raise ValueError(f"{field} does not exactly match compiled model; extend/align first")
    if not np.array_equal(np.asarray(traj.info.model.jnt_type), model.jnt_type):
        raise ValueError("Joint types differ")
    q0 = np.asarray(traj.data.qpos)
    q = _array(q0).copy()
    if q.ndim != 2 or q.shape[1] != model.nq or not np.isfinite(q).all():
        raise ValueError("Invalid qpos")
    split = np.asarray(traj.data.split_points, dtype=int)
    if (split.ndim != 1 or len(split) < 2 or split[0] != 0 or split[-1] != len(q)
        or np.any(np.diff(split) < 2)):
        raise ValueError("Every clip needs at least two frames and valid split points")
    frequency = float(traj.info.frequency)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Invalid trajectory frequency")
    root, floor, gids = _selected_box_geoms(model, foot_geom_names, root_joint_name, floor_name)
    z = int(model.jnt_qposadr[root]) + 2
    root_quat = q[:, z+1:z+5]
    if np.max(np.abs(np.linalg.norm(root_quat, axis=1)-1)) > 1e-3:
        raise ValueError("Input root quaternions are not normalized; do not silently repair them here")
    data = mj.MjData(model)
    gaps_before = np.empty((len(q), len(gids)))
    for t, pose in enumerate(q):
        data.qpos[:] = pose
        mj.mj_kinematics(model, data)
        gaps_before[t] = box_clearances(model, data, floor, gids)
    offsets = np.zeros(len(q))
    h = gaps_before.min(axis=1)
    for a, b in zip(split[:-1], split[1:]):
        offsets[a:b] = clearance - (h[a:b].min() if mode == "global" else h[a:b])
    if np.max(np.abs(offsets)) > max_shift:
        raise ValueError(f"Required shift exceeds {max_shift} m: inspect model, scale, and selected geoms")
    q[:, z] += offsets

    # Preserve the original storage precision and check the value that is returned.
    out_q = q.astype(q0.dtype, copy=False)
    keep = np.arange(model.nq) != z
    if not np.array_equal(out_q[:, keep], q0[:, keep]):
        raise AssertionError("Grounding changed a non-root-Z coordinate")
    q = _array(out_q)
    qvel = np.zeros((len(q), model.nv))
    dt = 1.0/frequency
    for a, b in zip(split[:-1], split[1:]):
        edges = np.empty((b-a-1, model.nv))
        for t in range(a, b-1):
            mj.mj_differentiatePos(model, edges[t-a], dt, q[t], q[t+1])
        qvel[a], qvel[b-1] = edges[0], edges[-1]
        if b-a > 2:
            qvel[a+1:b-1] = 0.5*(edges[:-1]+edges[1:])
    cached = {"xpos": np.empty((len(q), model.nbody, 3)),
              "xquat": np.empty((len(q), model.nbody, 4)),
              "cvel": np.empty((len(q), model.nbody, 6)),
              "subtree_com": np.empty((len(q), model.nbody, 3)),
              "site_xpos": np.empty((len(q), model.nsite, 3)),
              "site_xmat": np.empty((len(q), model.nsite, 9))}
    gaps_after = np.empty_like(gaps_before)
    for t, pose in enumerate(q):
        data.qpos[:] = pose
        data.qvel[:] = qvel[t]
        mj.mj_forward(model, data)  # Evaluate the state, never integrate it.
        gaps_after[t] = box_clearances(model, data, floor, gids)
        for field in cached:
            cached[field][t] = getattr(data, field)
    mins = gaps_after.min(axis=1)
    for a, b in zip(split[:-1], split[1:]):
        if mode == "per_frame":
            err = np.abs(mins[a:b]-clearance).max()
        else:
            err = abs(mins[a:b].min()-clearance)
        if err > 2e-6:
            raise AssertionError(f"Grounding read-back error: {err} m")
    use_numpy = isinstance(traj.data.qpos, np.ndarray)
    if use_numpy:
        backend = np
    else:
        import jax.numpy as backend
    updates = {"qpos": backend.asarray(out_q), "qvel": backend.asarray(qvel)}
    updates.update({k: backend.asarray(v) for k, v in cached.items()})
    new_data = traj.data.replace(**updates) if hasattr(traj.data, "replace") else replace(traj.data, **updates)
    result = replace(traj, data=new_data)
    if not np.array_equal(np.asarray(result.data.qpos)[:, keep], q0[:, keep]):
        raise AssertionError("Backend conversion changed non-root-Z coordinates")
    summary = {"mode": mode, "geoms": list(foot_geom_names),
               "before_min_m": float(h.min()), "before_median_m": float(np.median(h)),
               "after_min_m": float(mins.min()), "after_median_m": float(np.median(mins)),
               "after_max_m": float(mins.max()),
               "max_abs_root_shift_m": float(np.abs(offsets).max()),
               "max_non_root_z_qpos_change": 0.0}
    return result, summary


def extend_motion(
    env_name: str,
    env_params: DictConfig,
    traj: Trajectory,
    logger: logging.Logger | None = None,
) -> Trajectory:
    """Align qpos/qvel and rebuild all body/site reference arrays on the CPU.

    Use the plain conversion environment, not the PPO-wrapped imitation env.
    As in the earlier LocoMuJoCo pipeline, loading may reorder coordinates and
    resample, and playback may recenter root X/Y. This is not a new fitting run.
    """
    if "MjxMyoBimanualArm" in env_name:
        env_name = "MyoBimanualArm"
    elif "Mjx" in env_name:
        env_name = env_name.replace("Mjx", "")

    env_cls = Mujoco.registered_envs[env_name]
    env = env_cls(
        **env_params,
        th_params={"random_start": False, "fixed_start_conf": (0, 0)},
    )
    original_frequency = float(traj.info.frequency)
    target_frequency = 1.0 / env.dt

    # Restore the original interpolation -> load -> forward/replay pipeline.
    traj_data, traj_info = interpolate_trajectories(
        traj.data, traj.info, target_frequency
    )
    aligned = replace(traj, data=traj_data, info=traj_info)
    env.load_trajectory(aligned, warn=False)
    aligned = env.th.traj

    callback = ExtendTrajData(
        env, model=env._model, n_samples=int(aligned.data.n_samples)
    )

    # Select each clip explicitly; a fixed start of (0, 0) otherwise replays
    # the first clip on every reset in the earlier LocoMuJoCo implementation.
    for trajectory_index in range(int(env.th.n_trajectories)):
        env.th.fixed_start_conf = (trajectory_index, 0)
        env.play_trajectory(
            n_episodes=1, render=False, callback_class=callback
        )

    traj_data, traj_info = callback.extend_trajectory_data(
        aligned.data, aligned.info
    )
    result = replace(aligned, data=traj_data, info=traj_info)

    # This test assumes these are the actual supporting collision boxes.
    result, report = ground_final_reference(
        result,
        env._model,
        root_joint_name=env.root_free_joint_xml_name,
        foot_geom_names=("foot_box_l", "foot_box_r"),
        mode="per_frame",      # Only for the no-flight walking test.
        clearance=0.0,
    )

    print("GROUNDING CHECK:", report)

    # Diagnostic acceptance criteria, not MuJoCo contact tolerances.
    assert report["max_non_root_z_qpos_change"] == 0.0
    assert abs(report["after_min_m"]) < 1e-4
    assert abs(report["after_median_m"]) < 1e-4
    assert abs(report["after_max_m"]) < 1e-4

    validate_reference_data(result)

    if logger is not None:
        logger.info(
            "Extended reference %.3f -> %.3f Hz: cvel=%s, subtree_com=%s",
            original_frequency,
            float(result.info.frequency),
            result.data.cvel.shape,
            result.data.subtree_com.shape,
        )
    return result


def create_multi_trajectory_hash(names: list[str]) -> str:
    """
    Generates a stable hash for a list of strings using SHA256 with incremental updates.

    Args:
        names (list[str]): The list of strings to hash.

    Returns:
        str: A hexadecimal hash string.
    """

    # Sort the list to ensure order invariance
    sorted_names = sorted(names)

    hash_obj = hashlib.sha256()
    for s in sorted_names:
        hash_obj.update(s.encode())
    return hash_obj.hexdigest()


def _load_trajectories_individually(
    env_name: str,
    dataset_list: list[str],
    robot_conf: DictConfig,
    logger,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
) -> Trajectory:
    """Load motions independently and keep source names aligned."""
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()
    # MJX and MuJoCo share retargeted motion caches.
    cache_env_name = env_name.replace("Mjx", "") if "Mjx" in env_name else env_name

    # Retargeting methods use separate cache namespaces.
    if retargeting_method == "gmr":
        path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, cache_env_name, "gmr")
    else:
        path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, cache_env_name)
    os.makedirs(path_robot_smpl_data, exist_ok=True)

    all_trajectories = []

    for i, single_dataset in enumerate(dataset_list):
        cache_path = os.path.join(path_robot_smpl_data, f"{single_dataset}.npz")

        if os.path.exists(cache_path) and not clear_cache:
            logger.info(
                f"Dataset {i + 1}/{len(dataset_list)}: Found existing retargeted motion file at {cache_path}. Loading ..."
            )
            # Keep cached arrays on the host while combining motions.
            trajectory = Trajectory.load(cache_path, backend=np)
            all_trajectories.append(trajectory)
            continue

        method_name = retargeting_method.upper() if retargeting_method else "SMPL"
        action = "Re-retargeting (clear_cache)" if clear_cache and os.path.exists(cache_path) else "Retargeting"
        logger.info(f"Dataset {i + 1}/{len(dataset_list)}: {action} AMASS motion file using {method_name} ...")

        try:
            loaded = load_retargeted_amass_trajectory_set(
                env_name,
                single_dataset,
                robot_conf,
                retargeting_method=retargeting_method,
                gmr_config=gmr_config,
                clear_cache=clear_cache,
            )
            all_trajectories.append(loaded)
        except Exception as e:
            logger.error(f"Skipping dataset '{single_dataset}' due to failure during retargeting: {e}")
            continue

    if not all_trajectories:
        raise RuntimeError(f"Retargeting produced no valid {env_name} trajectories.")
    if len(all_trajectories) == 1:
        return all_trajectories[0]
    else:
        logger.info(
            "Concatenating %d trajectories ...",
            len(all_trajectories),
        )
        traj_datas = [t.data for t in all_trajectories]
        traj_infos = [t.info for t in all_trajectories]
        logger.info(f"Concatenating {len(all_trajectories)} final {env_name} trajectories ...")
        traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos, backend=np)
        loaded = Trajectory(traj_info, traj_data)
        logger.info("Final concatenation successful!")

    return loaded


def load_retargeted_amass_trajectory(
    env_name: str,
    dataset_name: str | list[str] | tuple[str, ...],
    robot_conf: DictConfig = None,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
) -> Trajectory:
    """Load retargeted AMASS trajectory data."""
    loaded = load_retargeted_amass_trajectory_set(
        env_name=env_name,
        dataset_name=dataset_name,
        robot_conf=robot_conf,
        retargeting_method=retargeting_method,
        gmr_config=gmr_config,
        clear_cache=clear_cache,
    )

    loaded = replace(
        loaded,
        data=loaded.data.to_jax(),
        info=replace(
            loaded.info,
            model=loaded.info.model.to_jax(),
        ),
    )

    # Make this timing boundary meaningful despite asynchronous dispatch.
    jax.block_until_ready(loaded.data)
    print("[AMASS] JAX conversion complete", flush=True)

    return loaded


def load_retargeted_amass_trajectory_set(
    env_name: str,
    dataset_name: str | list[str] | tuple[str, ...],
    robot_conf: DictConfig = None,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
) -> Trajectory:
    """Load retargeted AMASS trajectories with their source names."""
    logger = setup_logger("amass", identifier="[MuscleMimic AMASS Retargeting Pipeline]")

    # if robot_conf is not provided, load default one it from the YAML file
    if robot_conf is None:
        robot_conf = load_robot_conf_file(env_name)

    # Check retargeting method
    if retargeting_method is None:
        retargeting_method = robot_conf.get("retargeting_method", "smpl")
    if gmr_config is None:
        gmr_config = robot_conf.get("gmr_config", None)

    # Process datasets individually to avoid trajectory concatenation with different traj info
    if isinstance(dataset_name, list | tuple) and len(dataset_name) > 1:
        return _load_trajectories_individually(
            env_name,
            dataset_name,
            robot_conf,
            logger,
            retargeting_method=retargeting_method,
            gmr_config=gmr_config,
            clear_cache=clear_cache,
        )

    path_to_converted_amass_datasets = get_converted_amass_dataset_path()

    # Normalize environment name for cache consistency (MJX and non-MJX share same cache)
    cache_env_name = env_name.replace("Mjx", "") if "Mjx" in env_name else env_name

    # Use separate cache directories for different retargeting methods
    if retargeting_method == "gmr":
        path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, cache_env_name, "gmr")
    else:
        path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, cache_env_name)

    if not os.path.exists(path_robot_smpl_data):
        os.makedirs(path_robot_smpl_data, exist_ok=True)

    # load trajectory file(s)
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    all_trajectories = []
    for i, d_name in enumerate(dataset_name):
        d_path = os.path.join(path_robot_smpl_data, f"{d_name}.npz")
        cache_exists = os.path.exists(d_path)
        if not cache_exists or clear_cache:
            # Only check imports when we actually need to run retargeting
            check_optional_imports()
            path_to_smpl_model = get_smpl_model_path()
            smpl_optimized_shape_cache_path = os.path.join(
                path_to_converted_amass_datasets, cache_env_name, OPTIMIZED_SHAPE_FILE_NAME
            )

            # Cache validation for the SMPL-specific optimized shape file.
            if retargeting_method == "smpl":
                # Validate cached shape file compatibility
                smpl_shape_cache_valid = False
                if os.path.exists(smpl_optimized_shape_cache_path):
                    try:
                        # Load cached data and check site count compatibility
                        _shape_new, _scale, _smpl2robot_pos, smpl2robot_rot_mat, _offset_z, _height_scale = joblib.load(
                            smpl_optimized_shape_cache_path
                        )
                        cached_n_sites = smpl2robot_rot_mat.shape[0]

                        # Get expected site count from current environment
                        env_cls = Mujoco.registered_envs[env_name]
                        env = env_cls(
                            **robot_conf.env_params,
                            th_params={"random_start": False, "fixed_start_conf": (0, 0)},
                        )
                        expected_n_sites = len(env.sites_for_mimic)

                        if cached_n_sites == expected_n_sites:
                            smpl_shape_cache_valid = True
                            logger.info(f"Found valid SMPL optimized shape cache at {smpl_optimized_shape_cache_path}")
                        else:
                            logger.warning(
                                "SMPL optimized shape cache has "
                                f"{cached_n_sites} sites, but environment expects {expected_n_sites}. Regenerating..."
                            )
                            os.remove(smpl_optimized_shape_cache_path)
                    except Exception as e:
                        logger.warning(f"Failed to validate SMPL optimized shape cache: {e}. Regenerating...")
                        try:
                            os.remove(smpl_optimized_shape_cache_path)
                        except OSError:
                            print("Failed to remove invalid cache file.")

                if not smpl_shape_cache_valid:
                    logger.info("SMPL optimized shape cache not found or invalid, fitting a new one ...")
                    fit_smpl_shape(
                        env_name,
                        robot_conf,
                        path_to_smpl_model,
                        smpl_optimized_shape_cache_path,
                        logger,
                    )
            else:
                logger.info(
                    f"Using {retargeting_method.upper()} retargeting - skipping SMPL-only optimized shape cache setup"
                )

            # Choose retargeting method
            action = "Re-retargeting (clear_cache)" if clear_cache and cache_exists else "Retargeting"
            logger.info(
                f"Dataset {i + 1}/{len(dataset_name)}: {action} AMASS motion file using optimized body shape ..."
            )
            motion_data = load_amass_data(d_name)
            path_converted_shape = os.path.join(
                path_to_converted_amass_datasets, f"{cache_env_name}/{OPTIMIZED_SHAPE_FILE_NAME}"
            )
            trajectory, analysis = fit_smpl_motion(
                env_name,
                robot_conf,
                path_to_smpl_model,
                motion_data,
                path_converted_shape,
                logger,
            )

            logger.info("Using Mujoco to calculate other model-specific entities ...")
            trajectory = extend_motion(env_name, robot_conf.env_params, trajectory, logger)
            trajectory.save(d_path)

            # Save analysis data to file
            analysis_path = d_path.replace(".npz", "_analysis.npz")
            np.savez(analysis_path, **analysis)
            logger.info(f"Saved analysis to {analysis_path}")
            all_trajectories.append(trajectory)
        else:
            logger.info(
                f"Dataset {i + 1}/{len(dataset_name)}: Found existing retargeted motion file at {d_path}. Loading ..."
            )
            trajectory = Trajectory.load(d_path, backend=np)
            all_trajectories.append(trajectory)

    if len(all_trajectories) == 1:
        loaded = all_trajectories[0]
    else:
        logger.info("Concatenating trajectories ...")
        traj_datas = [t.data for t in all_trajectories]
        traj_infos = [t.info for t in all_trajectories]
        traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos)
        loaded = Trajectory(traj_info, traj_data)
        print("loaded:", loaded)

    logger.info("Trajectory data loaded!")

    return loaded


def retarget_traj_from_robot_to_robot(
    env_name_source: str,
    traj_source: Trajectory,
    env_name_target: str,
    robot_conf_source: DictConfig = None,
    robot_conf_target: DictConfig = None,
    path_to_fitted_motion_source: str | None = None,
):
    check_optional_imports()

    logger = setup_logger("amass", identifier="[LocoMuJoCo's Robot2Robot Retargeting Pipeline]")

    path_to_smpl_model = get_smpl_model_path()
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()

    # if robot_conf is not provided, load default one it from the YAML file
    if robot_conf_source is None:
        robot_conf_source = load_robot_conf_file(env_name_source)
    if robot_conf_target is None:
        robot_conf_target = load_robot_conf_file(env_name_target)

    path_source_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, env_name_source)
    path_target_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, env_name_target)

    traj_target = motion_transfer_robot_to_robot(
        env_name_source,
        robot_conf_source,
        traj_source,
        path_source_robot_smpl_data,
        env_name_target,
        robot_conf_target,
        path_target_robot_smpl_data,
        path_to_smpl_model,
        logger,
        path_to_fitted_motion_source,
    )

    return traj_target
