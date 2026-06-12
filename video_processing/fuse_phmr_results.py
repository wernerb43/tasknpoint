import os
import joblib
import numpy as np
import copy
import argparse
import pdb
import torch
import json
import re
import subprocess
import sys
import cv2
from pathlib import Path

SMPLX_PATH = os.path.join(os.environ.get("PROMPTHMR_DATA_ROOT", ""), "body_models", "smplx")

from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
# from prompt_hmr.utils.rotation_conversions import matrix_to_axis_angle
"""

Fuses results from the separate results.pkl files and also adds the camera transforms.

PrompthMR results files are .pkl files that contain dictionaries with the following keys:

each result: {
    - camera: {
        'pred_cam_R': T, 3, 3
        'pred_cam_T': T, 3
        'img_focal': float
        'img_center': list of length 2
    }
    - people: dict with people id as keys and the following keys:
        - track_id (same as people key)
        - frames: list of frame indices T
        - bboxes: list of bounding boxes T x 4
        - detected: boolean of whether person was detected or not T x 1
        - smplx_pose: T x 22 x 3, 3
        - smplx_transl: T x 3
        - smplx_betas: T x 10
        - smplx_cam: dict with the following keys:
            - rotmat: T x 55, 3, 3
            - pose: T x 75
            - shape: T x 10
            - trans: T x 3
            - contact: T x 6
            - static_conf_logits T x 6
        - smplx_world: dict with the following keys:
            - pose: T x 165
            - shape: T x 10
            - trans: T x 3

    - timings {}

    this should be the same for all the results files since they are in the same run of run_prompthmr_on_video.py
    - has_tracks: boolean
    - has_hps_cam: boolean
    - has_hps_world: boolean
    - has_slam: boolean
    - has_hands: boolean
    - has_2d_kpts: boolean
    - has_post_opt: boolean
    - spec_calib: dict with keys([0, 'first_frame', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 'avg_focal_length', 'min_focal_length', 'max_focal_length', 'std_focal_length', 'median_focal_length'])
    - contact_joint_ids: list of joint contact indices?
    - camera_world: dict with keys
        'pred_cam_R': T, 3, 3
        'pred_cam_T': T, 3
        'Rwc': T, 3, 3
        'Twc': T, 3
        'Rcw': T, 3, 3
        'Tcw': T, 3
        'img_focal': float
        'img_center': list of length 2
        'viz_scale': float
        'viz_center': list of length 2
    - start_frame: start frame index of video
    - end_frame: end frame index of video
}

example run command:
python scripts/fuse_results.py \
    --folder_name 09_18_2025_16_15_court5 \
    --video_name 09_18_2025_16_50_18_000_court5_SE_01_16_53_14_360

python scripts/fuse_results.py \
    --folder_name 10_16_2025_16_51_court2 \
    --video_name 09_18_2025_16_50_18_000_court5_SE_01_16_53_14_360

video 1:
python scripts/fuse_results.py \
    --folder_name 10_02_2025_08_00_court5 \
    --video_name 10_02_2025_08_10_35_000_court5_WSW_1_08_16_39_250 \
    --pkl_filename 'world_fuse.pkl' \
    --timestamps '/Volumes/ilona_seagate/tennis_dataset/full_dataset/10_02_2025_08_00_court5/10_02_2025_08_10_35_000_court5_WSW_1_08_16_39_250_timestamps.npy' \
    --force

video 2:
python scripts/fuse_results.py \
    --folder_name 10_02_2025_08_00_court5 \
    --video_name 10_02_2025_08_10_09_000_court3_E_1_08_16_04_526 \
    --pkl_filename 'world_fuse.pkl' \
    --timestamps '/Volumes/ilona_seagate/tennis_dataset/full_dataset/10_02_2025_08_00_court5/10_02_2025_08_10_09_000_court3_E_1_08_16_04_526_timestamps.npy' \
    --force

video N:

09_18_2025_16_15_court5/09_18_2025_17_25_43_000_court5_NE_01_17_27_11_168

TODO video:
python scripts/fuse_results.py \
    --folder_name 09_18_2025_16_15_court5 \
    --video_name 09_18_2025_16_32_04_000_court5_N_01_16_34_02_273 \
    --pkl_filename 'world_fuse.pkl' \
    --timestamps '/Volumes/ilona_seagate/tennis_dataset/full_dataset/09_18_2025_16_15_court5/09_18_2025_16_32_04_000_court5_N_01_16_34_02_273_timestamps.npy' \
    --force

python scripts/fuse_results.py \
    --folder_name 09_18_2025_16_15_court5 \
    --video_name 09_18_2025_17_25_35_000_court5_NW_01_17_26_16_546 \
    --pkl_filename 'world_fuse.pkl' \
    --timestamps '/Volumes/ilona_seagate/tennis_dataset/full_dataset/09_18_2025_16_15_court5/09_18_2025_17_25_35_000_court5_NW_01_17_26_16_546_timestamps.npy' \
    --force



"""

####################################### camera calibration stuff #######################################
def extract_key_frame(video_path, output_path, timestamp=0.0):
    """Extract a single frame from video at given timestamp."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vsync", "0",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Extracted frame to {output_path}")
    return output_path


def ensure_calibration_exists(
        folder_name,
        video_name,
        video_base_path,
        phmr_results_path,
        calibration_results_folder_name="calibration_results",
        camera_pose_recovery_dir=None,
        calib_filename="frame_first_court_annotations_calibration.json",
        log_file=None,
        calib_root=None,
        ):
    """
    Check if camera calibration exists; if not, run the calibration pipeline.

    Args:
        folder_name: Parent folder name
        video_name: Video name (without extension)
        video_base_path: Base path where videos are stored
        phmr_results_path: Path to first PromptHMR results file (for focal length)
        calib_root: Direct path to calibration results root (e.g. /mnt/datasets/calibration_results).
                    When set, overrides camera_pose_recovery_dir+calibration_results_folder_name
                    for the calibration JSON lookup and output, while camera_pose_recovery_dir
                    is still used to locate recover_camera_pose.py and court_snapshots.

    Returns:
        Path to calibration JSON file

    Raises:
        FileNotFoundError: If calibration cannot be created
    """
    # Construct expected calibration path
    # repo root is two levels up from this script (scripts/fuse_results.py -> prompthmr -> repo)
    if camera_pose_recovery_dir is None:
        camera_pose_recovery_dir = str(Path(__file__).resolve().parents[2] / "camera_pose_recovery")
    if calib_root:
        calibration_path = os.path.join(calib_root, folder_name, video_name, "frame_first", calib_filename)
    else:
        calibration_path = os.path.join(
            camera_pose_recovery_dir,
            calibration_results_folder_name,
            folder_name,
            video_name,
            "frame_first",
            calib_filename,
        )

    # If calibration exists, return it
    if os.path.exists(calibration_path):
        print(f"Found existing calibration: {calibration_path}")
        return calibration_path

    print(f"\n{'='*60}")
    print(f"Camera calibration not found for: {folder_name}/{video_name}")
    print(f"Starting calibration pipeline...")
    print(f"{'='*60}\n")

    # Step 1: Find the video file
    video_path = None
    for ext in ['.MOV', '.mov', '.mp4', '.MP4']:
        candidate = os.path.join(video_base_path, folder_name, f"{video_name}{ext}")
        if os.path.exists(candidate):
            video_path = candidate
            break

    if video_path is None:
        msg = f"Could not find video file for {folder_name}/{video_name} in {video_base_path}"
        if log_file:
            with open(log_file, "a") as lf:
                lf.write(f"[FAILED - no video] {folder_name}/{video_name}: {msg}\n")
        raise FileNotFoundError(msg)

    print(f"Found video: {video_path}")

    # Step 2: Extract first frame
    snapshots_dir = os.path.join(
        camera_pose_recovery_dir,
        "court_snapshots",
        folder_name,
        video_name
    )
    frame_path = os.path.join(snapshots_dir, "frame_first.png")

    if not os.path.exists(frame_path):
        print(f"Extracting first frame...")
        extract_key_frame(video_path, frame_path, timestamp=5.0)
    else:
        print(f"Frame already exists: {frame_path}")

    # Step 3: Run camera pose recovery (this will launch the annotator)
    if calib_root:
        output_dir = os.path.join(calib_root, folder_name, video_name)
    else:
        output_dir = os.path.join(
            camera_pose_recovery_dir,
            calibration_results_folder_name,
            folder_name,
            video_name
        )

    recover_script = os.path.join(camera_pose_recovery_dir, "recover_camera_pose.py")

    cmd = [
        "python", recover_script,
        "--image_path", frame_path,
        "--camera_model", "pinhole",
        "--output_dir", output_dir,
        "--video_path", video_path,
    ]

    # Add phmr_results if it exists
    if phmr_results_path and os.path.exists(phmr_results_path):
        cmd.extend(["--phmr_results", phmr_results_path])

    print(f"\nLaunching court annotator...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run the calibration script (this is interactive)
    result = subprocess.run(cmd, cwd=camera_pose_recovery_dir)

    if result.returncode != 0:
        raise RuntimeError(f"Camera pose recovery failed with return code {result.returncode}")

    # Check if calibration was created
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(
            f"Calibration was not created. Expected at: {calibration_path}"
        )

    print(f"\nCalibration created successfully: {calibration_path}")
    return calibration_path


#############################################################################################

def create_world4d_w_joints_no_camera_transform(results,
                            smplx,
                            start_frame=0,
                            chunk_start=None,
                            world1_to_world2=None,
                            timestamps=None,
                            global_time_start=None,
                            og_video_path=None,
                            total=None, step=1, first_person=False, device=None):

    """
    create world4d dict with smplx joints added in too, without GT camera calibration.
    Uses SLAM world coordinates directly; applies world1_to_world2 if provided.
    When world1_to_world2 is None, trans_world / joints_world are in SLAM world space.
    """

    # court bounds (only used when world1_to_world2 is provided)
    X_MIN, X_MAX = -3, 27
    Y_MIN, Y_MAX = -3, 14
    Z_MIN, Z_MAX = -1.0, 3

    if total is None:
        total = len(results["camera"]["pred_cam_R"])
    else:
        total = min(total, len(results["camera"]["pred_cam_R"]))

    if timestamps is not None:
        total = min(total, len(timestamps) - start_frame)

    world4d = {}

    # Pre-compute per-pid: O(1) frame lookup + batched SMPLX world joints
    _pid_list_all = list(results["people"].keys()) if not first_person else [next(iter(results["people"]))]
    _pid_frame_to_idx = {}
    _pid_joints_precomp = {}
    for _pid in _pid_list_all:
        _people = results["people"][_pid]
        _frames = _people["frames"]
        _pid_frame_to_idx[_pid] = {int(_f): _k for _k, _f in enumerate(_frames)}
        _N = len(_frames)
        if _N == 0:
            _pid_joints_precomp[_pid] = np.zeros((0, 25, 3), dtype=np.float32)
            continue
        _poses = _people["smplx_world"]["pose"]
        _shapes = _people["smplx_world"]["shape"]
        _transls = _people["smplx_world"]["trans"]
        _rotmats = axis_angle_to_matrix(torch.Tensor(_poses.reshape(_N, 55, 3)))
        with torch.no_grad():
            _pid_joints_precomp[_pid] = smplx(
                global_orient=_rotmats[:, :1].to(device),
                body_pose=_rotmats[:, 1:22].to(device),
                betas=torch.Tensor(_shapes).to(device),
                transl=torch.Tensor(_transls).to(device),
            ).body_joints.cpu().numpy()  # (N, 25, 3)

    for i in range(0, total, step):
        pose = []
        shape = []
        transl = []
        track_id = []
        joints = []
        joints_world = []
        transl_world = []
        pose_world = []

        pid_list = results["people"].keys() if not first_person else [next(iter(results["people"]))]

        for pid in pid_list:
            people = results["people"][pid]
            frames = people["frames"]
            _idx = _pid_frame_to_idx[pid].get(i)

            # checks if the person is in this frame
            if _idx is not None:
                in_frame = np.array([_idx])

                pose_i = people["smplx_world"]["pose"][in_frame]
                shape_i = people["smplx_world"]["shape"][in_frame]
                transl_i = people["smplx_world"]["trans"][in_frame]
                track_id_i = people["track_id"]

                if world1_to_world2 is not None:
                    trans_h = np.concatenate([transl_i, np.ones((transl_i.shape[0], 1))], axis=1)
                    trans_world_i = (world1_to_world2 @ trans_h.T).T[:, :3]

                    # Check bounds in tennis court coordinates
                    x, y, z = trans_world_i[0]
                    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX and Z_MIN <= z <= Z_MAX):
                        continue
                else:
                    # No court calibration: use SLAM world coords as-is
                    trans_world_i = transl_i

                pose.append(pose_i)
                shape.append(shape_i)
                transl.append(transl_i)
                track_id.append(track_id_i)
                transl_world.append(trans_world_i)

                joints_i = _pid_joints_precomp[pid][_idx]  # (25, 3), pre-computed

                joints.append(joints_i.reshape(-1, 25, 3))

                if world1_to_world2 is not None:
                    # Transform joints to court world coordinates
                    joints_h = np.concatenate([joints_i, np.ones((joints_i.shape[0], 1))], axis=1)
                    joints_world_i = (world1_to_world2 @ joints_h.T).T[:, :3]

                    # Compute pose_world: rotate global orient into court coords
                    from prompt_hmr.utils.rotation_conversions import matrix_to_axis_angle
                    global_orient_rotmat = axis_angle_to_matrix(torch.Tensor(pose_i[:, :3]))  # (1, 3, 3)
                    R_world = torch.Tensor(world1_to_world2[:3, :3])
                    global_orient_world = R_world @ global_orient_rotmat[0]
                    global_orient_world_aa = matrix_to_axis_angle(global_orient_world.unsqueeze(0))
                    pose_world_i = copy.deepcopy(pose_i)
                    pose_world_i[:, :3] = global_orient_world_aa.numpy()
                else:
                    joints_world_i = joints_i
                    pose_world_i = pose_i

                joints_world.append(joints_world_i.reshape(-1, 25, 3))
                pose_world.append(pose_world_i)

        # Camera: store SLAM camera-to-world for this frame
        camera = np.eye(4)
        if "camera_world" in results and "Rcw" in results["camera_world"]:
            camera[:3, :3] = results["camera_world"]["Rcw"][i]
            camera[:3, 3] = results["camera_world"]["Tcw"][i]

        global_timestamp = (global_time_start + timestamps[i+start_frame]) if (timestamps is not None and global_time_start is not None) else None

        if len(track_id) > 0:
            world4d[i+start_frame] = {
                "pose": torch.tensor(np.concatenate(pose)).float().reshape(len(track_id), -1, 3), # N, 55, 3
                "pose_world": torch.tensor(np.concatenate(pose_world)).float().reshape(len(track_id), -1, 3), # N, 55, 3
                "shape": torch.tensor(np.concatenate(shape)).float(), # N, 10
                "trans": torch.tensor(np.concatenate(transl)).float(), # N, 3
                "track_id": torch.tensor(np.array(track_id)) - 1, # N
                "camera": camera, # 4, 4
                "joints": torch.tensor(np.concatenate(joints)).float().reshape(len(track_id), -1, 3), # N, 25, 3
                "joints_world": torch.tensor(np.concatenate(joints_world)).float().reshape(len(track_id), -1, 3), # N, 25, 3
                "trans_world": torch.tensor(np.concatenate(transl_world)).float(), # N, 3
                "phmrworld_to_tennisworld": world1_to_world2, # 4, 4 or None
                "relative_timestamp": timestamps[i+start_frame] if timestamps is not None else None,
                "global_timestamp": global_timestamp,
                "global_time_start": global_time_start,
                "og_rgb_frame": i + start_frame,
                "og_video_path": og_video_path,
                "chunk_start": chunk_start,
            }
        else:
            world4d[i+start_frame] = {
                "track_id": np.array([]),
                "camera": camera,
                "relative_timestamp": timestamps[i+start_frame] if timestamps is not None else None,
                "global_timestamp": global_timestamp,
                "global_time_start": global_time_start,
                "og_rgb_frame": i + start_frame,
                "og_video_path": og_video_path,
                "chunk_start": chunk_start,
            }

    return world4d

import copy
import numpy as np
import torch
import pdb

# TODO - move this into a utils file
def create_world4d_w_joints(results,
                            smplx,
                            start_frame=0,
                            chunk_start=None,
                            world1_to_world2=None,
                            R_calib_c2w=None,
                            T_calib_c2w=None,
                            K_intrinsics=None,
                            K_calib=None,
                            timestamps=None,
                            global_time_start=None,
                            og_video_path=None,
                            total=None, step=1, first_person=False, device=None,
                            world_bounds=None):

    """
    creates a dictionary w/ keys as frame indices of the predictions
    copied over from the prompthmr method, but adding in smplx joint transforms
    and our court calibration coords

    TODO - add timestamp info in too
    """

    # court bounds — expanded by 3m on each side to accommodate monocular depth
    # uncertainty from models like GENMO that use an identity camera world
    X_MIN, X_MAX = -6, 30
    Y_MIN, Y_MAX = -6, 17
    Z_MIN, Z_MAX = -1.0, 3

    if total is None:
        total = len(results["camera"]["pred_cam_R"])
    else:
        total = min(total, len(results["camera"]["pred_cam_R"]))

    if timestamps is not None:
        total = min(total, len(timestamps) - start_frame)

    world4d = {}

    # Pre-compute per-pid: O(1) frame lookup + batched SMPLX world and camera joints
    _pid_list_all = list(results["people"].keys()) if not first_person else [next(iter(results["people"]))]
    _pid_frame_to_idx = {}
    _pid_joints_world_precomp = {}
    _pid_joints_cam_precomp = {}
    _pid_root_cam_precomp = {}
    for _pid in _pid_list_all:
        _people = results["people"][_pid]
        _frames = _people["frames"]
        _pid_frame_to_idx[_pid] = {int(_f): _k for _k, _f in enumerate(_frames)}
        _N = len(_frames)
        if _N == 0:
            _pid_joints_world_precomp[_pid] = np.zeros((0, 25, 3), dtype=np.float32)
            _pid_joints_cam_precomp[_pid] = np.zeros((0, 25, 3), dtype=np.float32)
            _pid_root_cam_precomp[_pid] = np.zeros((0, 3), dtype=np.float32)
            continue
        # Batch world joints
        _poses = _people["smplx_world"]["pose"]
        _shapes = _people["smplx_world"]["shape"]
        _transls = _people["smplx_world"]["trans"]
        _rotmats = axis_angle_to_matrix(torch.Tensor(_poses.reshape(_N, 55, 3)))
        with torch.no_grad():
            _pid_joints_world_precomp[_pid] = smplx(
                global_orient=_rotmats[:, :1].to(device),
                body_pose=_rotmats[:, 1:22].to(device),
                betas=torch.Tensor(_shapes).to(device),
                transl=torch.Tensor(_transls).to(device),
            ).body_joints.cpu().numpy()  # (N, 25, 3)
        # Batch camera joints
        _rotmats_cam = _people["smplx_cam"]["rotmat"]   # (N, 55, 3, 3)
        _shapes_cam  = _people["smplx_cam"]["shape"]    # (N, 10)
        _transls_cam = _people["smplx_cam"]["trans"]    # (N, 3)
        with torch.no_grad():
            _out_cam = smplx(
                global_orient=torch.tensor(_rotmats_cam[:, :1]).float().to(device),
                body_pose=torch.tensor(_rotmats_cam[:, 1:22]).float().to(device),
                betas=torch.tensor(_shapes_cam).float().to(device),
                transl=torch.tensor(_transls_cam).float().to(device),
            )
        _pid_joints_cam_precomp[_pid] = _out_cam.body_joints.cpu().numpy()   # (N, 25, 3)
        _pid_root_cam_precomp[_pid]   = _out_cam.joints[:, 0, :].cpu().numpy()  # (N, 3)

    for i in range(0, total, step):
        pose = []
        shape = []
        transl = []
        track_id = []
        joints = []
        joints_world = []
        transl_world = []
        pose_world = []

        pid_list = results["people"].keys() if not first_person else [next(iter(results["people"]))]

        for pid in pid_list:
            people = results["people"][pid]
            frames = people["frames"]
            _idx = _pid_frame_to_idx[pid].get(i)

            # checks if the person is in this frame
            if _idx is not None:
                in_frame = np.array([_idx])

                pose_i = people["smplx_world"]["pose"][in_frame]
                shape_i = people["smplx_world"]["shape"][in_frame]
                transl_i = people["smplx_world"]["trans"][in_frame]
                track_id_i = people["track_id"]

                if world1_to_world2 is not None:
                    trans_h = np.concatenate([transl_i, np.ones((transl_i.shape[0], 1))], axis=1)
                    trans_world_i = (world1_to_world2 @ trans_h.T).T[:, :3]

                    # Check bounds in tennis court coordinates
                    x, y, z = trans_world_i[0]
                    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX and Z_MIN <= z <= Z_MAX):
                        continue

                pose.append(pose_i)
                shape.append(shape_i)
                transl.append(transl_i)
                track_id.append(track_id_i)

                joints_i    = _pid_joints_world_precomp[pid][_idx]  # (25, 3), pre-computed
                joints_cam_i = _pid_joints_cam_precomp[pid][_idx]   # (25, 3), pre-computed
                raw_root_cam = _pid_root_cam_precomp[pid][_idx]     # (3,),    pre-computed

                # --- START RAY-CASTING DEPTH CORRECTION ---
                # 1. Find 2D pixel of the true pelvis using PromptHMR intrinsics (K_intrinsics)
                uvw = K_intrinsics @ raw_root_cam
                u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]

                # 2. Calculate the correct Z-depth using the focal length ratio
                f_x_phmr = K_intrinsics[0, 0]
                f_x_calib = K_calib[0, 0]
                z_new = raw_root_cam[2] * (f_x_calib / f_x_phmr)

                # 3. Ray-cast back to 3D using true calibration (K_calib) to find correct pelvis location
                c_x_calib, c_y_calib = K_calib[0, 2], K_calib[1, 2]
                f_y_calib = K_calib[1, 1]

                x_new = (u - c_x_calib) * z_new / f_x_calib
                y_new = (v - c_y_calib) * z_new / f_y_calib
                true_root_cam = np.array([x_new, y_new, z_new])

                # 4. Shift the entire OpenPose body to the true root (preserving metric size)
                local_joints = joints_cam_i - raw_root_cam
                joints_cam_corrected = local_joints + true_root_cam
                # --- END RAY-CASTING DEPTH CORRECTION ---

                # Transform corrected camera-space joints to court world coordinates
                joints_world_i = (R_calib_c2w @ joints_cam_corrected.T + T_calib_c2w.reshape(3, 1)).T  # 25, 3

                joints.append(joints_i.reshape(-1, 25, 3))
                transl_world.append(trans_world_i)
                joints_world.append(joints_world_i.reshape(-1, 25, 3))

                global_orient_rotmat = axis_angle_to_matrix(torch.Tensor(pose_i[:, :3]))  # (1, 3, 3)
                R_world = torch.Tensor(world1_to_world2[:3, :3])  # just the rotation part (3, 3)
                global_orient_world = R_world @ global_orient_rotmat[0]  # (3, 3)

                # Assuming matrix_to_axis_angle is imported (from prompt_hmr.utils.rotation_conversions usually)
                from prompt_hmr.utils.rotation_conversions import matrix_to_axis_angle
                global_orient_world_aa = matrix_to_axis_angle(global_orient_world.unsqueeze(0))  # (1, 3)

                pose_world_i = copy.deepcopy(pose_i)
                pose_world_i[:,:3] = global_orient_world_aa.numpy()
                pose_world.append(pose_world_i)

        # Camera: store as 4x4 camera-to-world matrix
        camera = np.eye(4)
        camera[:3, :3] = R_calib_c2w
        camera[:3, 3] = T_calib_c2w

        if global_time_start is not None and global_time_start > 100000000000:
            pdb.set_trace()

        if len(track_id) > 0:
            global_timestamp = (global_time_start + timestamps[i+start_frame]) if (timestamps is not None and global_time_start is not None) else None

            if global_timestamp is not None and global_timestamp > 100000000000:
                pdb.set_trace()

            world4d[i+start_frame] = {
                "pose": torch.tensor(np.concatenate(pose)).float().reshape(len(track_id), -1, 3), # N, 55, 3
                "pose_world": torch.tensor(np.concatenate(pose_world)).float().reshape(len(track_id), -1, 3), # N, 55, 3
                "shape": torch.tensor(np.concatenate(shape)).float(), # N, 10
                "trans": torch.tensor(np.concatenate(transl)).float(), # N, 3
                "track_id": torch.tensor(np.array(track_id)) - 1, # N
                "camera": camera, # 4, 4
                "K": K_intrinsics, # 3, 3 K_phmr scaled to original video resolution
                "K_calib": K_calib, # 3, 3 K from court calibration
                "joints": torch.tensor(np.concatenate(joints)).float().reshape(len(track_id), -1, 3), # N, 25, 3
                "joints_world": torch.tensor(np.concatenate(joints_world)).float().reshape(len(track_id), -1, 3), # N, 25, 3
                "trans_world": torch.tensor(np.concatenate(transl_world)).float(), # N, 3
                "phmrworld_to_tennisworld": world1_to_world2, # 4, 4
                "relative_timestamp": timestamps[i+start_frame] if timestamps is not None else None,
                "global_timestamp": global_timestamp,
                "global_time_start": global_time_start,
                "og_rgb_frame": i + start_frame,
                "og_video_path": og_video_path,
                "chunk_start": chunk_start,
            }
        else:
            world4d[i+start_frame] = {
                "track_id": np.array([]),
                "camera": camera,
                "K": K_intrinsics,
                "K_calib": K_calib,
                "relative_timestamp": timestamps[i+start_frame] if timestamps is not None else None,
                "global_timestamp": (global_time_start + timestamps[i+start_frame]) if (timestamps is not None and global_time_start is not None) else None,
                "global_time_start": global_time_start,
                "og_rgb_frame": i + start_frame,
                "og_video_path": og_video_path,
                "chunk_start": chunk_start,
            }

    return world4d

def compute_world1_to_world2_transform(R1_c2w, T1_c2w, R2_c2w, T2_c2w, debug=False):

    # conver w2c --> c2w
    R1_w2c = R1_c2w.T
    C1 = T1_c2w
    t1_w2c = -R1_w2c @ C1

    world1_to_camera = np.eye(4)
    world1_to_camera[:3, :3] = R1_w2c
    world1_to_camera[:3, 3] = t1_w2c

    # R2 is already c2w
    camera_to_world2 = np.eye(4)
    camera_to_world2[:3, :3] = R2_c2w
    camera_to_world2[:3, 3] = T2_c2w

    # Compose
    world1_to_world2 = camera_to_world2 @ world1_to_camera

    return world1_to_world2

def get_max_results_end_frame(results_folder):
    """
    Parse results_START_END.pkl filenames in results_folder and return the maximum END frame index.
    Does not load any pkl files.
    Returns -1 if no matching files are found.
    """
    pkl_pattern = re.compile(r"results_(\d+)_(\d+)\.pkl")
    max_end = -1
    try:
        for f in os.listdir(results_folder):
            m = pkl_pattern.fullmatch(f)
            if m:
                end_frame = int(m.group(2))
                if end_frame > max_end:
                    max_end = end_frame
    except FileNotFoundError:
        pass
    return max_end


def get_results_file_list(results_folder):
    """
    Return sorted list of (start_frame, end_frame, filename) tuples for all results_*.pkl files.
    """
    pkl_pattern = re.compile(r"results_(\d+)_(\d+)\.pkl")
    files = []
    try:
        for f in os.listdir(results_folder):
            m = pkl_pattern.fullmatch(f)
            if m:
                files.append((int(m.group(1)), int(m.group(2)), f))
    except FileNotFoundError:
        pass
    return sorted(files, key=lambda x: x[0])


def load_results(results_folder, start_result_idx=0, crop_results=None, verbose=True):
    """
    Load and sort PromptHMR result chunks from a folder.

    Args:
        results_folder: Path to folder containing results_START_END.pkl files
        start_result_idx: Index to start loading from (default: 0)
        crop_results: Max number of files to load (default: None = load all)
        verbose: Print loading progress (default: True)

    Returns:
        List of loaded result dictionaries, sorted by start frame
    """
    pkl_pattern = re.compile(r"results_(\d+)_(\d+)\.pkl")

    def get_start_frame(filename):
        match = pkl_pattern.match(os.path.basename(filename))
        return int(match.group(1)) if match else float('inf')

    # Find and sort matching files
    results_files = [
        os.path.join(results_folder, f)
        for f in os.listdir(results_folder)
        if pkl_pattern.fullmatch(f)
    ]
    results_files.sort(key=get_start_frame)

    # Take subset if requested
    if crop_results is not None and crop_results < len(results_files):
        results_files = results_files[start_result_idx:start_result_idx + crop_results]

    if verbose:
        print(f"Processing {len(results_files)} result files")

    # Load all files
    results = []
    for results_file in results_files:
        if verbose:
            print(f"Loading: {os.path.basename(results_file)}")

        results.append(joblib.load(results_file))

    return results

def fuse_and_spatially_align(phmr_results, camera_pose_recovery_results, timestamps=None, global_time_start=None, og_video_path=None, debug=False, world_bounds=None, video_height=None, initial_world4d=None):

    # set up SMPLX model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx = SMPLX_Layer(SMPLX_PATH).to(device)

    world_4d_full = copy.deepcopy(initial_world4d) if initial_world4d is not None else {}

    # load the camera calibration parameters in tennis court coordinates
    # New format: R_c2w and t_c2w (camera-to-world)

    R_calib_c2w = np.array(camera_pose_recovery_results.get('R_c2w', None))
    T_calib_c2w = np.array(camera_pose_recovery_results.get('t_c2w', None))  # camera center in world

    # Load camera intrinsics matrix K from calibration (3x3)
    K_calib = np.array(camera_pose_recovery_results.get('K', None))

    if K_calib is not None:
        R_calib_c2w = R_calib_c2w.astype(np.float64)
        T_calib_c2w = T_calib_c2w.astype(np.float64)
        K_calib = K_calib.astype(np.float64)

    # Compute K_phmr scaled to original video resolution.
    # PromptHMR downscales video (max_height=896) so its intrinsics are for
    # a smaller image. We scale them to match the original video resolution
    # so that reprojection onto the original RGB frames is correct.
    phmr_focal = phmr_results[0]["camera"]["img_focal"]
    phmr_center = phmr_results[0]["camera"]["img_center"]
    K_phmr = np.eye(3, dtype=np.float64)
    K_phmr[0, 0] = phmr_focal
    K_phmr[1, 1] = phmr_focal
    K_phmr[0, 2] = phmr_center[0]
    K_phmr[1, 2] = phmr_center[1]

    model_has_intrinsics = phmr_focal != 0.0 and phmr_center[1] != 0.0

    if not model_has_intrinsics:
        # Model (e.g. GVHMR) did not populate img_focal / img_center.
        # Fall back to K_calib so downstream ray-casting is a no-op (f_x_phmr == f_x_calib).
        K_phmr_scaled = K_calib.copy() if K_calib is not None else np.eye(3, dtype=np.float64)
        print("\nWarning: model intrinsics are zero (GVHMR-style), using K_calib as K_intrinsics")
    elif video_height is not None:
        # Use separate x/y scale factors: int rounding during downscale
        # (pipeline/utils.py:_process_frame_dimensions) means scale_x != scale_y.
        phmr_height = phmr_center[1] * 2  # PromptHMR center = half of processing height
        phmr_width = phmr_center[0] * 2
        scale_y = video_height / phmr_height
        # We don't have video_width here, so compute it from the aspect ratio
        # video_width = video_height * (phmr_width / phmr_height) approximately
        # But the rounding means we should use the actual video width if available.
        # For now, scale_x = scale_y is close enough (differs by <0.1%).
        scale_x = scale_y
        K_phmr_scaled = K_phmr.copy()
        K_phmr_scaled[0, 0] *= scale_x
        K_phmr_scaled[1, 1] *= scale_y
        K_phmr_scaled[0, 2] *= scale_x
        K_phmr_scaled[1, 2] *= scale_y
        print(f"\nK_phmr scaled to original video resolution (scale_y={scale_y:.4f}):\n{K_phmr_scaled}")
    else:
        # Fallback: use K_phmr unscaled (will be wrong for full-res projection)
        K_phmr_scaled = K_phmr
        print("\nWarning: video_height not provided, K_phmr not scaled to original resolution")

    print(f"K_calib (from court calibration):\n{K_calib}")

    for idx, result in enumerate(phmr_results):

        # load the PromptHMR calculated camera parametrs
        R_phmr = result['camera_world']['Rwc'][0]
        T_phmr = result['camera_world']['Twc'][0]

        # compute the transform from PromptHMR world to tennis court world
        world1_to_world2 = compute_world1_to_world2_transform(R_phmr,
                                                              T_phmr,
                                                              R_calib_c2w,
                                                              T_calib_c2w,
                                                              debug=True
                                                              )

        # create world4d with joints in tennis court world coordinates
        # using the loaded camera calibration parameters
        curr_world4d = create_world4d_w_joints(result,
                                               smplx,
                                               world1_to_world2=world1_to_world2,
                                               R_calib_c2w=R_calib_c2w,
                                               T_calib_c2w=T_calib_c2w,
                                               K_intrinsics=K_phmr_scaled,
                                               K_calib=K_calib,
                                               first_person=False,
                                               start_frame=result['start_frame'],
                                               chunk_start=result['start_frame'],
                                               timestamps=timestamps,
                                               global_time_start=global_time_start,
                                               og_video_path=og_video_path,
                                               world_bounds=world_bounds,
                                               device=device,
                                               )

        # create combined world4d with PID remapping
        if idx == 0 and not world_4d_full:
            world_4d_full = copy.deepcopy(curr_world4d)
        else:
            world_4d_full = fuse_worlds(world_4d_full, curr_world4d, debug=False)

    print("full world created")
    return world_4d_full


def fuse_and_spatially_align_no_camera_calib(phmr_results, timestamps=None, global_time_start=None, og_video_path=None, debug=False, initial_world4d=None):
    """
    Same as fuse_and_spatially_align but without GT camera calibration.
    Uses SLAM world coordinates directly (no court coordinate transform).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx = SMPLX_Layer(SMPLX_PATH).to(device)

    world_4d_full = copy.deepcopy(initial_world4d) if initial_world4d is not None else {}

    for idx, result in enumerate(phmr_results):
        curr_world4d = create_world4d_w_joints_no_camera_transform(
            result,
            smplx,
            world1_to_world2=None,
            first_person=False,
            start_frame=result['start_frame'],
            chunk_start=result['start_frame'],
            timestamps=timestamps,
            global_time_start=global_time_start,
            og_video_path=og_video_path,
            device=device,
        )

        if idx == 0 and not world_4d_full:
            world_4d_full = copy.deepcopy(curr_world4d)
        else:
            world_4d_full = fuse_worlds(world_4d_full, curr_world4d, debug=False)

    print("full world created (no camera calib)")
    return world_4d_full


def fuse_worlds(world_4d_full, curr_world4d, debug=False):
    """Merge curr_world4d into world_4d_full with PID remapping."""

    # Get PID mapping: curr_pid -> full_pid
    pid_mapping = get_pid_ordering(world_4d_full, curr_world4d, debug=debug)
    if debug:
        print(f"PID mapping: {pid_mapping}")

    # Assign new PIDs to unmatched curr people
    next_pid = _get_max_pid(world_4d_full) + 1
    curr_pids = _get_all_pids(curr_world4d)

    for cpid in curr_pids:
        if cpid not in pid_mapping:
            pid_mapping[cpid] = next_pid
            next_pid += 1
            if debug:
                print(f"  New person: curr {cpid} -> full {pid_mapping[cpid]}")

    if debug:
        print(f"Final mapping: {pid_mapping}")

    # Merge frames
    overlap_frames = set(world_4d_full.keys()) & set(curr_world4d.keys())

    for frame_idx, curr_data in curr_world4d.items():
        if len(curr_data.get('track_id', [])) == 0:
            if frame_idx not in world_4d_full:
                world_4d_full[frame_idx] = copy.deepcopy(curr_data)
            continue

        # Get current PIDs and compute remapped PIDs
        curr_pids = curr_data['track_id'].numpy().astype(int)
        remapped_pids = np.array([pid_mapping[p] for p in curr_pids])

        if frame_idx in overlap_frames and frame_idx in world_4d_full:
            # Overlap: add new people only
            full_data = world_4d_full[frame_idx]
            if len(full_data.get('track_id', [])) == 0:
                new_data = _reorder_frame_data(curr_data, remapped_pids)
                world_4d_full[frame_idx] = new_data
            else:
                existing_pids = set(full_data['track_id'].numpy().astype(int))
                new_indices = [i for i, p in enumerate(remapped_pids) if p not in existing_pids]

                if new_indices:
                    _append_people(full_data, curr_data, new_indices,
                                   [remapped_pids[i] for i in new_indices])
        else:
            # New frame: copy with remapped PIDs AND reordered data
            new_data = _reorder_frame_data(curr_data, remapped_pids)
            world_4d_full[frame_idx] = new_data

    return world_4d_full


def _reorder_frame_data(frame_data, remapped_pids):
    """
    Reorder frame data so that the N dimension is sorted by remapped PID.
    This ensures consistent ordering: person with lowest PID is at index 0, etc.
    """
    new_data = copy.deepcopy(frame_data)

    # Get sort order: indices that would sort remapped_pids in ascending order
    sort_order = np.argsort(remapped_pids)
    sorted_pids = remapped_pids[sort_order]

    # Reorder all tensor fields along the N dimension
    for key in ['pose', 'shape', 'trans', 'joints', 'joints_world', 'trans_world']:
        if key in new_data and isinstance(new_data[key], torch.Tensor):
            new_data[key] = new_data[key][sort_order]

    # Set the sorted track_ids
    new_data['track_id'] = torch.tensor(sorted_pids)

    return new_data


def _append_people(full_frame, curr_frame, indices, new_pids):
    """Append specific people from curr to full frame."""
    for key in ['pose', 'shape', 'trans', 'joints', 'joints_world', 'trans_world']:
        if key in full_frame and key in curr_frame:
            full_frame[key] = torch.cat([
                full_frame[key],
                curr_frame[key][indices]
            ], dim=0)

    full_frame['track_id'] = torch.cat([
        full_frame['track_id'],
        torch.tensor(new_pids)
    ])

    # Re-sort the entire frame by track_id to maintain consistent ordering
    _sort_frame_by_pid(full_frame)


def _sort_frame_by_pid(frame_data):
    """Sort all data in frame by track_id (in place)."""
    if len(frame_data.get('track_id', [])) == 0:
        return

    track_ids = frame_data['track_id'].numpy()
    sort_order = np.argsort(track_ids)

    for key in ['pose', 'shape', 'trans', 'joints', 'joints_world', 'trans_world']:
        if key in frame_data and isinstance(frame_data[key], torch.Tensor):
            frame_data[key] = frame_data[key][sort_order]

    frame_data['track_id'] = frame_data['track_id'][sort_order]


def _get_max_pid(world4d):
    max_pid = -1
    for frame_data in world4d.values():
        if len(frame_data.get('track_id', [])) > 0:
            max_pid = max(max_pid, int(frame_data['track_id'].max()))
    return max_pid


def _get_all_pids(world4d):
    pids = set()
    for frame_data in world4d.values():
        if len(frame_data.get('track_id', [])) > 0:
            pids.update(frame_data['track_id'].numpy().astype(int).tolist())
    return pids


def _append_people(full_frame, curr_frame, indices, new_pids):
    """Append specific people from curr to full frame."""
    for key in ['pose', 'shape', 'trans', 'joints', 'joints_world', 'trans_world']:
        if key in full_frame and key in curr_frame:
            full_frame[key] = torch.cat([
                full_frame[key],
                curr_frame[key][indices]
            ], dim=0)

    full_frame['track_id'] = torch.cat([
        full_frame['track_id'],
        torch.tensor(new_pids)
    ])

def get_pid_ordering(world_4d_full, curr_world4d, threshold=2.0, min_overlap=5, debug=False):
    """
    Compute mapping: curr_pid -> full_pid based on trans_world distances.

    Returns:
        mapping: dict {curr_track_id: full_track_id} for matched people
    """

    overlapping_frames = sorted(set(world_4d_full.keys()) & set(curr_world4d.keys()))

    if len(overlapping_frames) < min_overlap:
        return {}

    # Collect observations: {track_id: {frame: trans_world}}
    def collect_observations(world4d, frames):
        obs = {}
        for f in frames:
            frame_data = world4d[f]
            if len(frame_data.get('track_id', [])) == 0:
                continue

            track_ids = frame_data['track_id'].numpy().astype(int)
            trans_world = frame_data['trans_world'].numpy()

            for i, tid in enumerate(track_ids):
                if tid not in obs:
                    obs[tid] = {}
                obs[tid][f] = trans_world[i]  # (3,)

        return obs

    full_obs = collect_observations(world_4d_full, overlapping_frames)
    curr_obs = collect_observations(curr_world4d, overlapping_frames)

    if debug:
        print(f"Full PIDs: {list(full_obs.keys())}")
        print(f"Curr PIDs: {list(curr_obs.keys())}")

        # pdb.set_trace()

    # Compute pairwise distances over COMMON frames only
    candidates = []  # (curr_pid, full_pid, mean_dist, num_common_frames)

    for curr_pid, curr_frames in curr_obs.items():
        for full_pid, full_frames in full_obs.items():
            # Find frames where BOTH people are visible
            common_frames = set(curr_frames.keys()) & set(full_frames.keys())

            if len(common_frames) < min_overlap:
                continue

            # Compute mean L2 distance
            distances = [
                np.linalg.norm(curr_frames[f] - full_frames[f])
                for f in common_frames
            ]
            mean_dist = np.mean(distances)

            candidates.append((curr_pid, full_pid, mean_dist, len(common_frames)))

            if debug:
                print(f"  curr {curr_pid} <-> full {full_pid}: dist={mean_dist:.3f}m, frames={len(common_frames)}")

    # Greedy assignment: sort by distance, assign non-conflicting pairs
    candidates.sort(key=lambda x: x[2])

    mapping = {}  # curr_pid -> full_pid
    used_full_pids = set()

    for curr_pid, full_pid, dist, n_frames in candidates:
        if dist > threshold:
            break
        if curr_pid in mapping or full_pid in used_full_pids:
            continue

        mapping[curr_pid] = full_pid
        used_full_pids.add(full_pid)

        if debug:
            print(f"  MATCHED: curr {curr_pid} -> full {full_pid} (dist={dist:.3f}m)")

    return mapping

def get_global_start_time_ms(video_name):
    """
    extract global start time in milliseconds from video title

    Video name format:
    {month}_{day}_{year}_{starthour}_{startminute}_{startsecond}_{startmillisecond}_court{number}_{direction}_{endhour}_{endminute}_{endsecond}_{endmillisecond}

    """

    parts = video_name.split('_')

    # Extract time components
    # parts[0] = month, parts[1] = day, parts[2] = year
    # parts[3] = start hour, parts[4] = start minute, parts[5] = start second, parts[6] = start millisecond
    start_hour = int(parts[3])
    start_minute = int(parts[4])

    if "overhead" in video_name:
        start_second = 0
        start_millisecond = 0
    else:
        start_second = int(parts[5])
        start_millisecond = int(parts[6])

    # Convert to milliseconds
    global_start_ms = (
        start_hour * 3600 * 1000 +      # hours to ms
        start_minute * 60 * 1000 +       # minutes to ms
        start_second * 1000 +            # seconds to ms
        start_millisecond                # already in ms
    )


    # if global_start_ms > 1758278414000:
    if global_start_ms > 100000000000:
        pdb.set_trace()

    return global_start_ms

def main(
    folder_name,
    video_name,
    start_result_idx,
    crop_results,
    pkl_filename,
    timestamps,
    camera_pose_recovery_dir,
    video_base_path="/Volumes/ilona_seagate/tennis_dataset/full_dataset",
    skip_calibration=False,
    no_camera_calib=False,
    calibration_results_folder_name="calibration_results",
    phmr_results_folder_name="results",
    calib_filename="frame_first_court_annotations_calibration.json",
    world_bounds=None,
    force=False,
    incremental=False,
    log_file=None,
    calib_root=None,
):
    """
    Main fusion function with automatic calibration if needed.

    Args:
        folder_name: Parent folder name
        video_name: Video name (without extension)
        start_result_idx: Start index for result files
        crop_results: Number of result files to process
        pkl_filename: Output filename
        timestamps: Path to timestamps file
        video_base_path: Base path where original videos are stored
        skip_calibration: If True, fail instead of running calibration
    """
    ###### load the results files ######
    results_folder = f"{phmr_results_folder_name}/{folder_name}/{video_name}"
    fused_file = os.path.join(results_folder, pkl_filename)

    # Check if output already exists
    initial_world4d = None
    if os.path.exists(fused_file) and not force:
        if not incremental:
            print(f"Output already exists: {fused_file}")
            print("Skipping. Use --force to overwrite.")
            return None

        # Incremental mode: check whether new results files have appeared
        max_results_end = get_max_results_end_frame(results_folder)
        existing_fused = joblib.load(fused_file)
        max_fused_key = max(existing_fused.keys()) if existing_fused else -1

        # Filename END is exclusive (results_0_500.pkl → keys 0..499, max_fused_key=499)
        if max_results_end - 1 <= max_fused_key:
            print(f"Already up to date (max fused frame={max_fused_key}, "
                  f"max results end={max_results_end}). Skipping.")
            return existing_fused

        print(f"Incremental update: max fused frame={max_fused_key}, "
              f"max results end={max_results_end}. Processing new chunks.")

        # Find start_result_idx: re-process from the last results file that
        # overlaps with the existing fused dict (for PID continuity).
        results_file_list = get_results_file_list(results_folder)
        overlap_idx = 0
        for i, (start, end, _) in enumerate(results_file_list):
            if start <= max_fused_key:
                overlap_idx = i
        start_result_idx = overlap_idx
        initial_world4d = existing_fused
        print(f"Re-processing from results file index {start_result_idx} "
              f"(start_frame={results_file_list[overlap_idx][0]})")

    phmr_results = load_results(
        results_folder,
        start_result_idx=start_result_idx,
        crop_results=crop_results
    )

    if len(phmr_results) == 0:
        print(f"No PromptHMR results found in {results_folder}")
        return None

    ###### No-calibration fast path ######
    if no_camera_calib:
        print("--no_camera_calib set: skipping camera calibration, using SLAM world coordinates")
        global_time_start = None
        if timestamps is not None:
            print(f"Loading timestamps from {timestamps}")
            timestamps = np.load(timestamps)
            global_time_start = get_global_start_time_ms(video_name)

        og_video_path = f"{folder_name}/{video_name}"
        fused_results = fuse_and_spatially_align_no_camera_calib(
            phmr_results=phmr_results,
            timestamps=timestamps,
            global_time_start=global_time_start,
            og_video_path=og_video_path,
            debug=True,
            initial_world4d=initial_world4d,
        )
        joblib.dump(fused_results, fused_file)
        print(f"\nFused results saved to {fused_file}")
        return fused_results

    ###### Check/create camera calibration ######
    # Get path to first results file for phmr_results reference
    pkl_pattern = re.compile(r"results_(\d+)_(\d+)\.pkl")
    first_results_file = None
    for f in sorted(os.listdir(results_folder)):
        if pkl_pattern.fullmatch(f):
            first_results_file = os.path.join(results_folder, f)
            break

    # repo root is two levels up from this script (scripts/fuse_results.py -> prompthmr -> repo)
    if not camera_pose_recovery_dir:
        camera_pose_recovery_dir = str(Path(__file__).resolve().parents[2] / "camera_pose_recovery")

    if calib_root:
        camera_pose_recovery_path = os.path.join(
            calib_root, folder_name, video_name, "frame_first", calib_filename
        )
    else:
        camera_pose_recovery_path = os.path.join(
            f"{camera_pose_recovery_dir}/{calibration_results_folder_name}",
            folder_name, video_name,
            "frame_first",
            calib_filename,
        )

    def _log_failure(reason):
        msg = f"[FAILED - {reason}] {folder_name}/{video_name}\n"
        print(msg, end="")
        if log_file:
            with open(log_file, "a") as lf:
                lf.write(msg)

    if not os.path.exists(camera_pose_recovery_path):
        if skip_calibration:
            print(f"Calibration not found and --skip_calibration is set: {camera_pose_recovery_path}")
            _log_failure("no calibration")
            return None

        # Run calibration pipeline
        try:
            camera_pose_recovery_path = ensure_calibration_exists(
                folder_name=folder_name,
                video_name=video_name,
                video_base_path=video_base_path,
                phmr_results_path=first_results_file,
                camera_pose_recovery_dir=camera_pose_recovery_dir,
                calibration_results_folder_name=calibration_results_folder_name,
                calib_filename=calib_filename,
                log_file=log_file,
                calib_root=calib_root,
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Failed to create calibration: {e}")
            _log_failure(f"calibration error: {e}")
            return None

    ###### load the camera parameters from the calibration ######
    camera_pose_recovery_results = json.load(open(camera_pose_recovery_path))

    ###### load timestamps if provided ######
    global_time_start = None
    if timestamps is not None:
        print(f"Loading timestamps from {timestamps}")
        timestamps = np.load(timestamps)
        global_time_start = get_global_start_time_ms(video_name)

    ###### exit if no camera calibration is found ######
    if "R_c2w" not in camera_pose_recovery_results:
        print(f"Camera calibration not found in: {camera_pose_recovery_path}")
        return None

    ###### get original video resolution for K scaling ######
    video_height = None
    for ext in ['.MOV', '.mov', '.mp4', '.MP4']:
        candidate = os.path.join(video_base_path, folder_name, f"{video_name}{ext}")
        if os.path.exists(candidate):
            cap = cv2.VideoCapture(candidate)
            if cap.isOpened():
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Original video resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{video_height}")
                cap.release()
            break
    if video_height is None:
        print("Warning: could not determine original video resolution for K scaling")

    ###### fuse the results and add the camera transform ######
    # Build the original video path for RGB frame extraction
    og_video_path = f"{folder_name}/{video_name}"

    fused_results = fuse_and_spatially_align(
        phmr_results=phmr_results,
        camera_pose_recovery_results=camera_pose_recovery_results,
        timestamps=timestamps,
        global_time_start=global_time_start,
        og_video_path=og_video_path,
        debug=True,
        world_bounds=world_bounds,
        video_height=video_height,
        initial_world4d=initial_world4d,
    )

    ###### save the fused results ######
    joblib.dump(fused_results, fused_file)
    print(f"\nFused results saved to {fused_file}")

    return fused_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse consecutive PromptHMR result chunks into a single file."
    )

    parser.add_argument(
        "--folder_name",
        default="test_visualizations",
        help="Parent folder under results/ that contains the video folder.",
    )
    parser.add_argument(
        "--video_name",
        default="test_keypoint_video",
        help="Video folder inside results/<folder_name>/ containing the chunked pkl files.",
    )
    parser.add_argument(
        "--pkl_filename",
        default="world4d_fused.pkl",
        help="fused pkl filename to save the fused results as",
    )
    parser.add_argument(
        "--take_subset",
        action="store_true",
        help="Fuse only a subset of result files (see --start_result_idx and --crop_results).",
    )
    parser.add_argument(
        "--start_result_idx",
        type=int,
        default=0,
        help="Index to start from when --take_subset is enabled.",
    )
    parser.add_argument(
        "--crop_results",
        type=int,
        default=-1,
        help="Number of result files to fuse when --take_subset is enabled.",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        default=None,
        help="Path to timestamps file corresponding to the video.",
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        default="/Volumes/ilona_seagate/tennis_dataset/full_dataset",
        help="Base path where original videos are stored (for frame extraction).",
    )
    parser.add_argument(
        "--camera_pose_recovery_dir",
        default=None,
        help="Path to the camera_pose_recovery repo directory (contains recover_camera_pose.py). "
             "Defaults to <repo_root>/camera_pose_recovery."
    )
    parser.add_argument(
        "--calib_root",
        default=None,
        help="Direct path to the calibration results root directory "
             "(e.g. /mnt/datasets/calibration_results). When set, overrides "
             "camera_pose_recovery_dir + calibration_results_folder_name for "
             "calibration JSON lookup and output.",
    )
    parser.add_argument(
        "--calibration_results_folder_name",
        type=str,
        default="calibration_results",
        help="Folder name where camera pose recovery results are stored.",
    )
    parser.add_argument(
        "--phmr_results_folder_name",
        type=str,
        default="/Volumes/ilona_seagate/tennis_dataset/prompthmr_results/results",
        help="Folder name where PromptHMR results are stored.",
    )
    parser.add_argument(
        "--calib_filename",
        type=str,
        default="frame_first_court_annotations_calibration.json",
        help="Calibration JSON filename inside the frame_first/ directory.",
    )
    parser.add_argument(
        "--world_bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="World-coordinate bounds for filtering: x_min x_max y_min y_max z_min z_max (default: tennis court)",
    )
    parser.add_argument(
        "--skip_calibration",
        action="store_true",
        help="Skip videos that don't have calibration (don't launch annotator).",
    )
    parser.add_argument(
        "--no_camera_calib",
        action="store_true",
        help="Run fusion without GT camera calibration; outputs in SLAM world coordinates.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="If a fused file already exists, only process new results chunks and merge them in.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to a log file. Failed videos are appended here.",
    )
    args = parser.parse_args()

    # Handle force flag
    results_folder = f"/Volumes/ilona_seagate/tennis_dataset/prompthmr_results/results/{args.folder_name}/{args.video_name}"
    fused_file = os.path.join(results_folder, args.pkl_filename)
    if args.force and os.path.exists(fused_file):
        os.remove(fused_file)

    main(
        folder_name=args.folder_name,
        video_name=args.video_name,
        start_result_idx=args.start_result_idx,
        crop_results=args.crop_results if args.crop_results > 0 else None,
        pkl_filename=args.pkl_filename,
        timestamps=args.timestamps,
        video_base_path=args.video_base_path,
        skip_calibration=args.skip_calibration,
        no_camera_calib=args.no_camera_calib,
        calibration_results_folder_name=args.calibration_results_folder_name,
        phmr_results_folder_name=args.phmr_results_folder_name,
        camera_pose_recovery_dir=args.camera_pose_recovery_dir,
        calib_filename=args.calib_filename,
        world_bounds=tuple(args.world_bounds) if args.world_bounds else None,
        force=args.force,
        incremental=args.incremental,
        log_file=args.log_file,
        calib_root=args.calib_root,
    )
