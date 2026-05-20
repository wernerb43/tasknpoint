import argparse
import csv
import os
import pathlib
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

GMR_PATH = pathlib.Path(__file__).parent.parent / "submodules" / "GMR"
sys.path.insert(0, str(GMR_PATH))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer as _BaseRobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
import general_motion_retargeting.params as _gmr_params
from rich import print

INITIAL_ROBOT_HEIGHT = 0.79


class RobotMotionViewer(_BaseRobotMotionViewer):
    def __init__(self, *args, keyboard_callback=None, **kwargs):
        self._user_key_callback = keyboard_callback
        self.paused = False
        self.current_frame = 0
        self.total_frames = 0
        super().__init__(*args, keyboard_callback=self._key_callback, **kwargs)

    def _key_callback(self, keycode):
        if keycode == 32:  # space bar
            self.paused = not self.paused
            if self.paused:
                frame_time = self.current_frame / self.motion_fps
                print(f"[bold yellow]PAUSED[/bold yellow] at frame {self.current_frame}/{self.total_frames}  t={frame_time:.3f}s")
            else:
                print(f"[bold green]RESUMED[/bold green]")
        if self._user_key_callback is not None:
            self._user_key_callback(keycode)

    def step(self, *args, rate_limit=True, **kwargs):
        super().step(*args, rate_limit=False, **kwargs)
        while self.paused:
            self.viewer.sync()
            time.sleep(0.05)
        if rate_limit:
            self.rate_limiter.sleep()


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", type=str, default="")
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
            "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
            "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
            "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
            "tienkung", "fourier_gr3",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--robot_xml", type=str, default=None, help="Override the robot XML path")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--loop", default=False, action="store_true")
    parser.add_argument("--record_video", default=False, action="store_true")
    parser.add_argument("--rate_limit", default=False, action="store_true")
    # parser.add_argument("--cam_distance_scale", type=float, default=1.5)
    args = parser.parse_args()

    if args.robot_xml is not None:
        _gmr_params.ROBOT_XML_DICT[args.robot] = pathlib.Path(args.robot_xml)

    SMPLX_FOLDER = GMR_PATH / "assets" / "body_models"

    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )

    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    start_frame = max(0, min(args.start_frame, len(smplx_data_frames) - 1))

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",
        # cam_distance_scale=args.cam_distance_scale,
    )
    robot_motion_viewer.total_frames = len(smplx_data_frames) - start_frame

    # Compute offsets from start frame so robot begins at [0, 0, INITIAL_ROBOT_HEIGHT] with identity rotation
    first_frame_qpos = retarget.retarget(smplx_data_frames[start_frame])

    pos_offset = np.array([0.0, 0.0, INITIAL_ROBOT_HEIGHT]) - first_frame_qpos[:3]

    initial_quat_xyzw = first_frame_qpos[[4, 5, 6, 3]]  # wxyz -> xyzw
    offset_rot = R.from_quat(initial_quat_xyzw).inv()
    start_pos = first_frame_qpos[:3].copy()

    print(f"Calculated pos_offset: {pos_offset}")
    print(f"Calculated offset_rot: {offset_rot.as_euler('xyz', degrees=True)} deg")

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    i = start_frame
    while True:
        qpos = retarget.retarget(smplx_data_frames[i])

        rotated_delta = offset_rot.apply(qpos[:3] - start_pos)
        adjusted_root_pos = rotated_delta + np.array([0.0, 0.0, INITIAL_ROBOT_HEIGHT])

        root_rot_xyzw = qpos[[4, 5, 6, 3]]  # wxyz -> xyzw
        result_xyzw = (offset_rot * R.from_quat(root_rot_xyzw)).as_quat()
        adjusted_root_rot = result_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz

        rel = i - start_frame
        robot_motion_viewer.current_frame = rel
        print(f"\rFrame {rel:>5d}/{len(smplx_data_frames) - start_frame}  t={rel / aligned_fps:.3f}s", end="", flush=True)

        transformed_human_data = {
            body_name: (
                offset_rot.apply(pos - start_pos) + np.array([0.0, 0.0, INITIAL_ROBOT_HEIGHT]),
                (offset_rot * R.from_quat(rot[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]],
            )
            for body_name, (pos, rot) in retarget.scaled_human_data.items()
        }

        robot_motion_viewer.step(
            root_pos=adjusted_root_pos,
            root_rot=adjusted_root_rot,
            dof_pos=qpos[7:],
            human_motion_data=transformed_human_data,
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )

        if args.save_path is not None:
            adjusted_qpos = qpos.copy()
            adjusted_qpos[:3] = adjusted_root_pos
            adjusted_qpos[3:7] = adjusted_root_rot
            qpos_list.append(adjusted_qpos)

        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break

    if args.save_path is not None:
        root_pos = np.array([q[:3] for q in qpos_list])
        root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
        dof_pos = np.array([q[7:] for q in qpos_list])

        csv_path = (
            args.save_path.replace(".pkl", ".csv")
            if args.save_path.endswith(".pkl")
            else args.save_path + ".csv"
        )

        combined_data = np.hstack([root_pos, root_rot, dof_pos])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(combined_data)
        print(f"\nSaved to {csv_path}")

    robot_motion_viewer.close()
