import os
import joblib
import numpy as np
import copy
import argparse
import torch
import re
import sys
import pdb

# SMPLX_PATH derived from env rather than caltech-tennis-benchmark's data_config
SMPLX_PATH = os.path.join(os.environ["PROMPTHMR_DATA_ROOT"], "body_models", "smplx")

# Add PromptHMR model repo to sys.path so prompt_hmr.* imports resolve
_phmr = os.environ.get("PHMR_REPO", "")
if _phmr and _phmr not in sys.path:
    sys.path.insert(0, _phmr)

from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer

"""
Robo script fusion - simple version without timestamps or camera calibration.
Fuses result files into one combined world4d, preserving camera data from each chunk.
"""


def create_world4d_w_joints(results, smplx, start_frame=0, total=None, step=1, first_person=False, device=None):
    """
    Create world4d dict with smplx joints added (no camera transform).
    Copies camera data directly from results.
    """
    if total is None:
        total = len(results["camera_world"]["Rwc"])
    else:
        total = min(total, len(results["camera_world"]["Rwc"]))

    world4d = {}


    for i in range(0, total, step):
        pose = []
        shape = []
        transl = []
        track_id = []
        joints = []

        pid_list = results["people"].keys() if not first_person else [next(iter(results["people"]))]

        camera_w = results["camera_world"]
        Rwc = camera_w["Rwc"][i]
        Twc = camera_w["Twc"][i]
        camera = np.eye(4)
        camera[:3, :3] = Rwc
        camera[:3, 3] = Twc

        for pid in pid_list:
            people = results["people"][pid]
            frames = people["frames"]
            in_frame = np.where(frames == i)[0]

            # Check if the person is in this frame
            if len(in_frame) == 1:
                pose_i = people["smplx_world"]["pose"][in_frame]
                shape_i = people["smplx_world"]["shape"][in_frame]
                transl_i = people["smplx_world"]["trans"][in_frame]
                track_id_i = people["track_id"]

                pose.append(pose_i)
                shape.append(shape_i)
                transl.append(transl_i)
                track_id.append(track_id_i)

                # Compute smplx joints
                rotmat = axis_angle_to_matrix(torch.Tensor(pose_i.reshape(1, 55, 3)))
                with torch.no_grad():
                    joints_i = (
                        smplx(
                            global_orient=rotmat[:, :1].to(device),
                            body_pose=rotmat[:, 1:22].to(device),
                            betas=torch.Tensor(shape_i).to(device),
                            transl=torch.Tensor(transl_i).to(device),
                        )
                        .body_joints[0]
                        .cpu()
                        .numpy()
                    )  # 25, 3

                joints.append(joints_i.reshape(-1, 25, 3))

        if len(track_id) > 0:
            world4d[i + start_frame] = {
                "pose": torch.tensor(np.concatenate(pose)).float().reshape(len(track_id), -1, 3),  # N, 55, 3
                "shape": torch.tensor(np.concatenate(shape)).float(),  # N, 10
                "trans": torch.tensor(np.concatenate(transl)).float(),  # N, 3
                "track_id": torch.tensor(np.array(track_id)) - 1,  # N
                "joints": torch.tensor(np.concatenate(joints)).float().reshape(len(track_id), -1, 3),  # N, 25, 3
                "camera": camera,
            }
        else:
            world4d[i + start_frame] = {
                "track_id": np.array([]),
                "camera": camera,
            }

    return world4d


def load_results(results_folder, start_result_idx=0, crop_results=None, verbose=True):
    """
    Load and sort PromptHMR result chunks from a folder.
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


def fuse_results(phmr_results, debug=False):
    """
    Fuse PromptHMR results without spatial alignment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx = SMPLX_Layer(SMPLX_PATH).to(device)

    world_4d_full = {}

    for idx, result in enumerate(phmr_results):
        curr_world4d = create_world4d_w_joints(
            result,
            smplx,
            first_person=True,
            start_frame=result['start_frame'],
            device=device,
        )

        if idx == 0:
            world_4d_full = copy.deepcopy(curr_world4d)
        else:
            world_4d_full = fuse_worlds(world_4d_full, curr_world4d, debug=debug)

    return world_4d_full


def fuse_results_simple(phmr_results, debug=False):
    """
    Fuse PromptHMR results without any PID matching - just concatenate frames.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx = SMPLX_Layer(SMPLX_PATH).to(device)

    world_4d_full = {}

    for idx, result in enumerate(phmr_results):
        curr_world4d = create_world4d_w_joints(
            result,
            smplx,
            first_person=False,
            start_frame=result['start_frame'],
            device=device,
        )

        if idx == 0:
            world_4d_full = copy.deepcopy(curr_world4d)
        else:
            world_4d_full = fuse_worlds_simple(world_4d_full, curr_world4d, debug=debug)

    return world_4d_full


def fuse_worlds_simple(world_4d_full, curr_world4d, debug=False):
    """
    Simple merge: just add new frames from curr_world4d, skip overlapping frames.
    No PID remapping - keeps original track_ids from each chunk.
    """

    for frame_idx, curr_data in curr_world4d.items():

        if frame_idx not in world_4d_full:
            # New frame: just copy it
            world_4d_full[frame_idx] = copy.deepcopy(curr_data)
        elif debug:
            print(f"  Skipping overlapping frame {frame_idx}")

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
    Camera data is preserved as-is (not per-person).
    """
    new_data = copy.deepcopy(frame_data)

    # Get sort order: indices that would sort remapped_pids in ascending order
    sort_order = np.argsort(remapped_pids)
    sorted_pids = remapped_pids[sort_order]

    # Reorder all tensor fields along the N dimension (per-person data)
    for key in ['pose', 'shape', 'trans', 'joints']:
        if key in new_data and isinstance(new_data[key], torch.Tensor):
            new_data[key] = new_data[key][sort_order]

    # Set the sorted track_ids
    new_data['track_id'] = torch.tensor(sorted_pids)

    # Camera data is preserved automatically via deepcopy

    return new_data


def _append_people(full_frame, curr_frame, indices, new_pids):
    """Append specific people from curr to full frame."""
    for key in ['pose', 'shape', 'trans', 'joints']:
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
    """Sort all data in frame by track_id (in place). Camera data is unchanged."""
    if len(frame_data.get('track_id', [])) == 0:
        return

    track_ids = frame_data['track_id'].numpy()
    sort_order = np.argsort(track_ids)

    # Only sort per-person data, camera stays as-is
    for key in ['pose', 'shape', 'trans', 'joints']:
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


def get_pid_ordering(world_4d_full, curr_world4d, threshold=2.0, min_overlap=5, debug=False):
    """
    Compute mapping: curr_pid -> full_pid based on trans distances.
    """
    overlapping_frames = sorted(set(world_4d_full.keys()) & set(curr_world4d.keys()))

    if len(overlapping_frames) < min_overlap:
        return {}

    # Collect observations: {track_id: {frame: trans}}
    def collect_observations(world4d, frames):
        obs = {}
        for f in frames:
            frame_data = world4d[f]
            if len(frame_data.get('track_id', [])) == 0:
                continue

            track_ids = frame_data['track_id'].numpy().astype(int)
            trans = frame_data['trans'].numpy()

            for i, tid in enumerate(track_ids):
                if tid not in obs:
                    obs[tid] = {}
                obs[tid][f] = trans[i]  # (3,)

        return obs

    full_obs = collect_observations(world_4d_full, overlapping_frames)
    curr_obs = collect_observations(curr_world4d, overlapping_frames)

    if debug:
        print(f"Full PIDs: {list(full_obs.keys())}")
        print(f"Curr PIDs: {list(curr_obs.keys())}")

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


def main(
    folder_name,
    video_name,
    start_result_idx,
    crop_results,
    pkl_filename,
    simple=False,
):
    """
    Main fusion function - simple version without camera calibration or timestamps.

    Args:
        simple: If True, use simple fusion without PID matching.
    """
    # Load the results files
    results_folder = f"{folder_name}/{video_name}"
    fused_file = os.path.join(results_folder, pkl_filename)

    # Check if output already exists
    if os.path.exists(fused_file):
        print(f"Output already exists: {fused_file}")
        print("Skipping. Use --force to overwrite.")
        return None

    phmr_results = load_results(
        results_folder,
        start_result_idx=start_result_idx,
        crop_results=crop_results
    )

    if len(phmr_results) == 0:
        print(f"No PromptHMR results found in {results_folder}")
        return None

    # Fuse the results
    if simple:
        print("Using simple fusion (no PID matching)")
        fused_results = fuse_results_simple(phmr_results, debug=True)
    else:
        fused_results = fuse_results(phmr_results, debug=True)

    # Save the fused results
    joblib.dump(fused_results, fused_file)
    print(f"\nFused results saved to {fused_file}")

    return fused_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse consecutive PromptHMR result chunks into a single file (no camera calibration)."
    )
    parser.add_argument(
        "--folder_name",
        default="/Volumes/ilona_seagate/tennis_dataset/prompthmr_results/robo_results",
        help="Parent folder under results/ that contains the video folder.",
    )
    parser.add_argument(
        "--video_name",
        default="02_11_2026_16_19_04_000_court1_X_1_16_19_52_396",
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
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple fusion without PID matching (just concatenate frames).",
    )
    args = parser.parse_args()

    # Handle force flag
    results_folder = f"{args.folder_name}/{args.video_name}"
    fused_file = os.path.join(results_folder, args.pkl_filename)
    if args.force and os.path.exists(fused_file):
        os.remove(fused_file)

    main(
        folder_name=args.folder_name,
        video_name=args.video_name,
        start_result_idx=args.start_result_idx,
        crop_results=args.crop_results if args.crop_results > 0 else None,
        pkl_filename=args.pkl_filename,
        simple=args.simple,
    )
