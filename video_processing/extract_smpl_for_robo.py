import os
import sys
import cv2
import numpy as np
import torch
import time
import tyro
import joblib
import pdb
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


SMPLX_PATH = os.path.join(os.environ.get("PROMPTHMR_DATA_ROOT", ""), "body_models", "smplx")

from fuse_phmr_results import create_world4d_w_joints_no_camera_transform

from prompt_hmr.utils.rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix
# from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix

# from prompthmr.vis.viser import viser_vis_world4d_no_video, get_color

from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer

import os.path as osp

"""

def main(
    results_pkl_path: str,
    video_path: str = "",
    has_world_coords=False,
    is_og_pkl_file=False,
    run_viser=True,
    start_frame = 0,
    end_frame = 176,
    out_folder="/Users/ilonademler/Documents/research/human_motion/caltech-tennis-benchmark/prompthmr/robo/robo_results/robo_formatted_outputs",
    action_title="tennis_forehand",
    timestamps_path=None,
):

script for extacting SMPL parameters to be passed into robot retargeting pipeline.

currently extracts ball throws (imperfect)

example usage:

left step catch:

python extract_smpl_for_robo.py \
    --results-pkl-path /Volumes/ilona_seagate/tennis_dataset/prompthmr_results/robo_results/02_03_2026_19_00_ath/02_03_2026_19_20_00_000_court1_X_1_19_24_22_083/results_5500_5999.pkl \
    --timestamps_path /Volumes/ilona_seagate/tennis_dataset/robo_dataset/02_03_2026_19_00_ath/02_03_2026_19_20_00_000_court1_X_1_19_24_22_083_timestamps.npy \
    --timestamp_start 425 --timestamp_end 445 \
    --is_og_pkl_file \
    --start_frame 368 \
    --end_frame 482 \
    --out_folder /Volumes/ilona_seagate/tennis_dataset/prompthmr_results/robo_results/robo_formatted_outputs \
    --action_title left_step_catch

kick:

python extract_smpl_for_robo.py \
    --results-pkl-path /Volumes/ilona_seagate/tennis_dataset/prompthmr_results/robo_results/03_23_2026_11_00_chenlawn/03_23_2026_11_36_41_000_kick_X_2_11_38_48_425/results_3000_3499.pkl \
    --is_og_pkl_file \
    --start_frame 6 \
    --end_frame 164 \
    --out_folder /Volumes/ilona_seagate/tennis_dataset/prompthmr_results/robo_results/robo_formatted_outputs \
    --action_title kick2 \
    --run_viser \
    --video_path /Volumes/ilona_seagate/tennis_dataset/robo_dataset/03_23_2026_11_00_chenlawn/03_23_2026_11_36_41_000_kick_X_2_11_38_48_425.MOV

middle_catch:

python extract_smpl_for_robo.py \
    --results-pkl-path /Volumes/ilona_seagate/tennis_dataset/prompthmr_results/robo_results/03_23_2026_11_00_chenlawn/03_23_2026_11_34_08_000_centerthrow_X_1_11_35_43_460/results_500_999.pkl \
    --is_og_pkl_file \
    --start_frame 121 \
    --end_frame 276 \
    --out_folder /Volumes/ilona_seagate/tennis_dataset/prompthmr_results/robo_results/robo_formatted_outputs \
    --action_title middle_catch \
    --run_viser \
    --video_path /Volumes/ilona_seagate/tennis_dataset/robo_dataset/03_23_2026_11_00_chenlawn/03_23_2026_11_34_08_000_centerthrow_X_1_11_35_43_460.MOV

two step backhand:

python extract_smpl_for_robo.py \
    --results-pkl-path /mnt/datasets/robo_results/prompthmr_results/05_14_2026_09_00_gatesthomas/05_14_2026_09_00_34_000_court1_X_2_09_02_32_258/results_1000_1499.pkl \
    --is_og_pkl_file \
    --start_frame 0 \
    --end_frame 200 \
    --out_folder /mnt/datasets/robo_results/retarget_inputs \
    --action_title backhand_two_step_test \
    --run_viser \
    --video_path /Volumes/ilona_seagate/tennis_dataset/robo_dataset/05_14_2026_09_00_gatesthomas/05_14_2026_09_00_34_000_court1_X_2_09_02_32_258.MOV

one-handed-baseball:
python extract_smpl_for_robo.py \
    --results-pkl-path /mnt/datasets/robo_results/prompthmr_results/05_09_2026_13_00_CAST/05_09_2026_13_44_59_000_court1_X_1_13_45_15_265/results.pkl \
    --is_og_pkl_file \
    --start_frame 85 \
    --end_frame 157 \
    --out_folder /mnt/datasets/robo_results/retarget_inputs \
    --action_title one_hand_baseball_hit \
    --run_viser \
    --video_path /mnt/datasets/robodataset/05_09_2026_13_00_CAST/05_09_2026_13_44_59_000_court1_X_1_13_45_15_265.MOV


python extract_smpl_for_robo.py \
    --results-pkl-path /mnt/datasets/robo_results/prompthmr_results/05_14_2026_09_00_gatesthomas/05_14_2026_09_02_44_000_court1_X_3_09_05_52_755/results_4000_4499.pkl \
    --is_og_pkl_file \
    --start_frame 347 \
    --end_frame 499 \
    --out_folder /mnt/datasets/robo_results/retarget_inputs \
    --action_title soccer_kick_far_left \
    --run_viser \
    --video_path /mnt/datasets/robodataset/05_14_2026_09_00_gatesthomas/05_14_2026_09_02_44_000_court1_X_3_09_05_52_755.MOV


pickup
python extract_smpl_for_robo.py \
    --results-pkl-path /mnt/datasets/robo_results/prompthmr_results/05_18_2026_08_00_chesterave/05_18_2026_19_32_39_000_court1_X_5_19_33_12_365/results_250_749.pkl \
    --is_og_pkl_file \
    --start_frame 0 \
    --end_frame 66 \
    --out_folder /mnt/datasets/robo_results/retarget_inputs \
    --action_title pickup_test \
    --run_viser \
    --video_path /mnt/datasets/robodataset/05_18_2026_08_00_chesterave/05_18_2026_19_32_39_000_court1_X_5_19_33_12_365.MOV

new forehand:
python extract_smpl_for_robo.py \
    --results-pkl-path /mnt/datasets/robo_results/prompthmr_results/05_18_2026_08_00_chesterave/05_18_2026_19_32_39_000_court1_X_5_19_33_12_365/results_250_749.pkl \
    --is_og_pkl_file \
    --start_frame 67 \
    --end_frame 169 \
    --out_folder /mnt/datasets/robo_results/retarget_inputs \
    --action_title pickup_test \
    --run_viser \
    --video_path /mnt/datasets/robodataset/05_18_2026_08_00_chesterave/05_18_2026_19_32_39_000_court1_X_5_19_33_12_365.MOV

blake demo

11_23_2025_11_38_22_000_court5_SE_2_11_46_45_833
* 2220 - 2326

python extract_smpl_for_robo.py \
    --results-pkl-path /mnt/datasets/calten_evals/prompthmr_results/11_23_2025_10_00_court4/11_23_2025_11_38_22_000_court5_SE_2_11_46_45_833/results_1867_1977.pkl \
    --is_og_pkl_file \
    --start_frame 0 \
    --end_frame 110 \
    --out_folder /mnt/datasets/robo_results/retarget_inputs \
    --action_title blake_on_caltech_courts \
    --run_viser \
    --video_path /mnt/datasets/robodataset/11_23_2025_10_00_court4/11_23_2025_11_38_22_000_court5_SE_2_11_46_45_833.MOV


"""

import time
import copy
import numpy as np
import viser
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from vis_tools import checkerboard_geometry


def get_color(idx):
    """Get a color for a track ID."""
    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
    ]
    return colors[idx % len(colors)]


def get_floor_mesh_z_up(pred_vert_gr, z_offset=0.0, scale=1.5, floor_color=None):
    """Return the geometry of the floor mesh for Z-up coordinate system."""
    verts = pred_vert_gr.clone()

    # Scale of the scene using X and Y (floor plane when Z is up)
    sx, sy = (verts.max(0)[0].max(0)[0] - verts.min(0)[0].min(0)[0])[[0, 1]]
    scale = max(sx.item(), sy.item()) * scale

    # Center X
    cx = (verts.mean(1).max(0)[0] + verts.mean(1).min(0)[0])[[0]] / 2.0
    cx = cx.item()

    # Center Y
    cy = (verts.mean(1).max(0)[0] + verts.mean(1).min(0)[0])[[1]] / 2.0
    cy = cy.item()

    if floor_color is None:
        v, f, vc, fc = checkerboard_geometry(length=scale, c1=cx, c2=cy, up="z")
    else:
        v, f, vc, fc = checkerboard_geometry(
            length=scale,
            c1=cx,
            c2=cy,
            up="z",
            color0=floor_color[0],
            color1=floor_color[1],
        )

    # Apply z offset to floor
    v[:, 2] += z_offset
    vc = vc[:, :3] * 255

    return [v, f, vc]


def viser_vis_world4d_no_video_z_up(
    world4d,
    faces,
    init_fps=25,
    block=False,
    floor=None,
    img_maxsize=320,
):
    """Viser visualization for world4d data with Z-up coordinate system."""
    try:
        server.scene.reset()
    except NameError:
        server = viser.ViserServer()

    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")  # Z is up

    # Add axis labels
    axis_label_distance = 1.2  # Distance from origin for labels
    server.scene.add_label("/axis_label_x", text="X", position=(axis_label_distance, 0, 0))
    server.scene.add_label("/axis_label_y", text="Y", position=(0, axis_label_distance, 0))
    server.scene.add_label("/axis_label_z", text="Z", position=(0, 0, axis_label_distance))

    num_frames = len(world4d)
    frame_keys = sorted(world4d.keys())

    gui_timestep = server.gui.add_slider(
        "Timestep",
        min=0,
        max=num_frames - 1,
        step=1,
        initial_value=0,
        disabled=True,
    )
    gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
    gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
    gui_playing = server.gui.add_checkbox("Playing", True)
    gui_framerate = server.gui.add_slider(
        "FPS", min=1, max=60, step=0.1, initial_value=init_fps
    )
    gui_framerate_options = server.gui.add_button_group(
        "FPS options", ("10", "20", "30", "60")
    )
    gui_show_camera = server.gui.add_checkbox("Show Camera", True)

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
            if current_timestep < len(frustum_nodes):
                frustum_nodes[current_timestep].visible = gui_show_camera.value
        prev_timestep = current_timestep
        server.flush()

    @gui_show_camera.on_update
    def _(_) -> None:
        current_timestep = gui_timestep.value
        if current_timestep < len(frustum_nodes):
            with server.atomic():
                frustum_nodes[current_timestep].visible = gui_show_camera.value
            server.flush()

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=vtf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes = []
    mesh_nodes = []
    frustum_nodes = []

    for idx, frame_key in enumerate(tqdm(frame_keys)):
        # Add base frame.
        frame_nodes.append(
            server.scene.add_frame(f"/frames/t{idx}", show_axes=False)
        )

        # Place meshes in the frame
        world3d = world4d[frame_key]
        track_id = world3d["track_id"]
        if len(track_id) > 0:
            vertices = world3d["vertices"]
            vertices = copy.deepcopy(vertices)
            for tid_idx, (tid, verts) in enumerate(zip(track_id, vertices)):
                mesh_nodes.append(
                    server.scene.add_mesh_simple(
                        name=f"/frames/t{idx}/human_{tid}",
                        vertices=verts,
                        faces=faces,
                        flat_shading=False,
                        wireframe=False,
                        color=get_color(int(tid)),
                    )
                )

        # Place the frustum.
        camera = world3d["camera"]
        quat = R.from_matrix(camera[:3, :3]).as_quat()
        quat = np.concatenate([quat[3:], quat[:3]])
        trans = camera[:3, 3]

        fov = 0.96
        frustum_nodes.append(server.scene.add_camera_frustum(
            f"/frames/t{idx}/frustum",
            fov=fov,
            line_width=1.5,
            color=(255, 127, 14),
            aspect=1.7,
            scale=0.4,
            wxyz=quat,
            position=trans,
        ))

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{idx}/frustum/axes",
            axes_length=0.3,
            axes_radius=0.02,
        )

    # Add floor
    if floor is not None:
        fv, ff = floor
        server.scene.add_mesh_simple(
            f"/floor",
            vertices=fv,
            faces=ff,
            flat_shading=False,
            wireframe=True,
            color=(50, 50, 50),
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    if block:
        while True:
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
            time.sleep(1.0 / gui_framerate.value)

    gui = [gui_playing, gui_timestep, gui_framerate, num_frames]
    return server, gui


def viser_vis_world4d_z_up(
    images,
    world4d,
    faces,
    init_fps=25,
    block=False,
    floor=None,
    img_maxsize=320,
):
    """Viser visualization for world4d data with Z-up coordinate system and video frames."""
    try:
        server.scene.reset()
    except NameError:
        server = viser.ViserServer()

    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")  # Z is up

    # Add axis labels
    axis_label_distance = 1.2
    server.scene.add_label("/axis_label_x", text="X", position=(axis_label_distance, 0, 0))
    server.scene.add_label("/axis_label_y", text="Y", position=(0, axis_label_distance, 0))
    server.scene.add_label("/axis_label_z", text="Z", position=(0, 0, axis_label_distance))

    num_frames = len(world4d)
    frame_keys = sorted(world4d.keys())

    gui_timestep = server.gui.add_slider(
        "Timestep",
        min=0,
        max=num_frames - 1,
        step=1,
        initial_value=0,
        disabled=True,
    )
    gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
    gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
    gui_playing = server.gui.add_checkbox("Playing", True)
    gui_framerate = server.gui.add_slider(
        "FPS", min=1, max=60, step=0.1, initial_value=init_fps
    )
    gui_framerate_options = server.gui.add_button_group(
        "FPS options", ("10", "20", "30", "60")
    )
    gui_show_camera = server.gui.add_checkbox("Show Camera", True)

    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
            if current_timestep < len(frustum_nodes):
                frustum_nodes[current_timestep].visible = gui_show_camera.value
        prev_timestep = current_timestep
        server.flush()

    @gui_show_camera.on_update
    def _(_) -> None:
        current_timestep = gui_timestep.value
        if current_timestep < len(frustum_nodes):
            with server.atomic():
                frustum_nodes[current_timestep].visible = gui_show_camera.value
            server.flush()

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=vtf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes = []
    mesh_nodes = []
    frustum_nodes = []

    for idx, frame_key in enumerate(tqdm(frame_keys)):
        frame_nodes.append(
            server.scene.add_frame(f"/frames/t{idx}", show_axes=False)
        )

        # Place meshes in the frame
        world3d = world4d[frame_key]
        track_id = world3d["track_id"]
        if len(track_id) > 0:
            vertices = world3d["vertices"]
            vertices = copy.deepcopy(vertices)
            for tid_idx, (tid, verts) in enumerate(zip(track_id, vertices)):
                mesh_nodes.append(
                    server.scene.add_mesh_simple(
                        name=f"/frames/t{idx}/human_{tid}",
                        vertices=verts,
                        faces=faces,
                        flat_shading=False,
                        wireframe=False,
                        color=get_color(int(tid)),
                    )
                )

        # Place the frustum with image.
        image = images[idx] if idx < len(images) else None
        camera = world3d["camera"]
        quat = R.from_matrix(camera[:3, :3]).as_quat()
        quat = np.concatenate([quat[3:], quat[:3]])
        trans = camera[:3, 3]

        print("\n \n trans: ", trans, "\n \n")

        if image is not None and max(image.shape) > img_maxsize:
            scale = img_maxsize / max(image.shape)
            image = cv2.resize(
                image, None, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        fov = 0.96
        aspect = image.shape[1] / image.shape[0] if image is not None else 1.7
        frustum_nodes.append(server.scene.add_camera_frustum(
            f"/frames/t{idx}/frustum",
            fov=fov,
            aspect=aspect,
            line_width=1.5,
            color=(255, 127, 14),
            scale=0.4,
            wxyz=quat,
            position=trans,
            image=image,
        ))

        server.scene.add_frame(
            f"/frames/t{idx}/frustum/axes",
            axes_length=0.3,
            axes_radius=0.02,
        )

    # Add floor
    if floor is not None:
        fv, ff = floor
        server.scene.add_mesh_simple(
            f"/floor",
            vertices=fv,
            faces=ff,
            flat_shading=False,
            wireframe=True,
            color=(50, 50, 50),
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    if block:
        while True:
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
            time.sleep(1.0 / gui_framerate.value)

    gui = [gui_playing, gui_timestep, gui_framerate, num_frames]
    return server, gui


def save_stick_figure_video(world4d, out_path, fps=30, frame_arrows=None):
    """Save a 3D stick figure MP4 of the motion, centered on the person with fixed axis ranges."""
    import imageio.v3 as iio

    SKELETON_PAIRS = [
        (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
        (0, 15), (0, 16), (15, 17), (16, 18)
    ]

    frame_keys = sorted(world4d.keys())

    # Collect joints for all frames with people (first person)
    joints_per_frame = {}
    for k in frame_keys:
        world3d = world4d[k]
        if len(world3d["track_id"]) == 0:
            continue
        joints = world3d["joints"]
        if hasattr(joints, 'numpy'):
            joints = joints.numpy()
        else:
            joints = np.array(joints)
        joints_per_frame[k] = joints[0]  # 25, 3

    if len(joints_per_frame) == 0:
        print("No joints found, skipping stick figure video")
        return

    all_joints = np.stack(list(joints_per_frame.values()))  # T, 25, 3

    # Auto-detect "up" axis: the axis with the largest mean within-frame joint range
    # (feet-to-head distance ~1.7m dominates over horizontal body width).
    per_frame_range = all_joints.max(axis=1) - all_joints.min(axis=1)  # T, 3
    mean_range = per_frame_range.mean(axis=0)  # 3
    up_axis = int(mean_range.argmax())
    ha, hb = [i for i in range(3) if i != up_axis]  # two horizontal axes
    axis_names = ['X', 'Y', 'Z']
    print(f"Auto-detected up axis: {axis_names[up_axis]} (ranges: {mean_range})")

    # Fixed axis limits centered on the person
    center_ha = (all_joints[:, :, ha].min() + all_joints[:, :, ha].max()) / 2
    center_hb = (all_joints[:, :, hb].min() + all_joints[:, :, hb].max()) / 2
    up_min = all_joints[:, :, up_axis].min()
    up_max = all_joints[:, :, up_axis].max()

    horiz_range = max(
        all_joints[:, :, ha].max() - all_joints[:, :, ha].min(),
        all_joints[:, :, hb].max() - all_joints[:, :, hb].min(),
    )
    vert_range = up_max - up_min

    horiz_pad = max(horiz_range * 0.3, 0.5)
    vert_pad = max(vert_range * 0.2, 0.3)

    half = horiz_range / 2 + horiz_pad
    xlim = (center_ha - half, center_ha + half)
    ylim = (center_hb - half, center_hb + half)
    zlim = (up_min - vert_pad, up_max + vert_pad)

    last_joints = all_joints[0]
    frames_out = []
    for k in frame_keys:
        if k in joints_per_frame:
            joints_3d = joints_per_frame[k]
            last_joints = joints_3d
        else:
            joints_3d = last_joints

        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel(axis_names[ha])
        ax.set_ylabel(axis_names[hb])
        ax.set_zlabel(axis_names[up_axis] + ' (up)')
        ax.set_title(f'Frame {k}')

        # Draw floor plane at the minimum of the up axis
        xx, yy = np.meshgrid(xlim, ylim)
        zz = np.full_like(xx, up_min - vert_pad / 2)
        ax.plot_surface(xx, yy, zz, alpha=0.15, color='gray')

        # Draw skeleton bones — map ha→plot-X, hb→plot-Y, up_axis→plot-Z
        for i, j in SKELETON_PAIRS:
            if i < len(joints_3d) and j < len(joints_3d):
                ax.plot(
                    [joints_3d[i, ha], joints_3d[j, ha]],
                    [joints_3d[i, hb], joints_3d[j, hb]],
                    [joints_3d[i, up_axis], joints_3d[j, up_axis]],
                    color='steelblue', linewidth=2,
                )

        # Draw joint dots
        ax.scatter(joints_3d[:, ha], joints_3d[:, hb], joints_3d[:, up_axis],
                   c='tomato', s=25, zorder=5, depthshade=False)

        # Draw facing arrows if provided
        if frame_arrows and k in frame_arrows:
            for arrow in frame_arrows[k]:
                o = arrow["origin"]
                d = arrow["direction"]
                length = arrow.get("length", 0.5)
                color = arrow.get("color", "red")
                ax.quiver(
                    o[ha], o[hb], o[up_axis],
                    d[ha] * length, d[hb] * length, d[up_axis] * length,
                    color=color, linewidth=2, arrow_length_ratio=0.3,
                )

        ax.view_init(elev=20, azim=-50)
        fig.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        frames_out.append(img)
        plt.close(fig)

    print(f"Saving stick figure video ({len(frames_out)} frames) to {out_path} ...")
    iio.imwrite(str(out_path), frames_out, fps=fps, codec='libx264', pixelformat='yuv420p')
    print(f"Saved stick figure video to {out_path}")


def get_action_segments(world4d, output_dir=None, apply_pca_transform=True):
    """
    Extract action segments from world4d based on left wrist motion (ball throwing).

    Args:
        world4d: Dictionary mapping frame indices to pose data with 'joints' key
        output_dir: Directory to save the wrist coordinate plots (optional)
        apply_pca_transform: If True, apply PCA to find axis of maximal motion

    Returns:
        start_and_end_action_times: List of [start, end] frame indices for each action
    """
    # Joint indices (3DPW format)
    right_wrist_idx = 4
    left_wrist_idx = 7

    # Extract wrist coordinates across all frames
    sorted_frame_indices = sorted(world4d.keys())
    right_wrist_coords = torch.stack([world4d[frame_idx]['joints'][:1, right_wrist_idx]
                                       for frame_idx in sorted_frame_indices]).squeeze()  # (T, 3)
    left_wrist_coords = torch.stack([world4d[frame_idx]['joints'][:1, left_wrist_idx].float()
                                      for frame_idx in sorted_frame_indices]).squeeze()  # (T, 3)

    # Convert to numpy for plotting and analysis
    right_wrist_np = right_wrist_coords.numpy()
    left_wrist_np = left_wrist_coords.numpy()
    frame_indices = np.array(sorted_frame_indices)

    # =====================
    # 1. Plot 3x2 grid of wrist coordinates
    # =====================
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    coord_names = ['X', 'Y', 'Z']

    for i, coord_name in enumerate(coord_names):
        # Right wrist (column 0)
        try:
            axes[i, 0].plot(frame_indices, right_wrist_np[:, i], 'b-', linewidth=0.8)
        except:
            pdb.set_trace()
        axes[i, 0].set_xlabel('Frame Index')
        axes[i, 0].set_ylabel(f'{coord_name} Coordinate (m)')
        axes[i, 0].set_title(f'Right Wrist - {coord_name} vs Frame')
        axes[i, 0].grid(True, alpha=0.3)

        # Left wrist (column 1)
        axes[i, 1].plot(frame_indices, left_wrist_np[:, i], 'r-', linewidth=0.8)
        axes[i, 1].set_xlabel('Frame Index')
        axes[i, 1].set_ylabel(f'{coord_name} Coordinate (m)')
        axes[i, 1].set_title(f'Left Wrist - {coord_name} vs Frame')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'wrist_coordinates.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Saved wrist coordinate plot to: {plot_path}")
    plt.close()

    # =====================
    # 2. Find axis of maximal motion using PCA (optional)
    # =====================
    if apply_pca_transform:
        # Use left wrist since that's the throwing arm
        right_centered = right_wrist_np - right_wrist_np.mean(axis=0)

        # PCA via SVD
        U, S, Vt = np.linalg.svd(right_centered, full_matrices=False)

        # Project onto principal component (axis of maximal variance)
        # Vt[0] is the first principal component direction
        motion_signal = right_centered @ Vt[0]  # Project onto PC1

        print(f"PCA principal direction: {Vt[0]}")
        print(f"Variance explained: {S**2 / (S**2).sum()}")
    else:
        # Default: use X-coordinate (assuming arm moves back/forward along X)
        motion_signal = right_wrist_np[:, 0]

    # =====================
    # 3. Segment cyclic motion by finding peaks (hand farthest back)
    # =====================
    # The throwing motion is cyclic - we want to find when the hand is at its
    # extreme position (farthest back before throwing)

    # Try both peaks and troughs, use the one that makes more sense
    min_distance = 30  # Minimum frames between actions (adjust based on motion speed)

    # Find peaks (maxima)
    peaks_pos, properties_pos = find_peaks(motion_signal, distance=min_distance, prominence=0.05)

    # Find troughs (minima) by negating signal
    peaks_neg, properties_neg = find_peaks(-motion_signal, distance=min_distance, prominence=0.05)

    # Use peaks with higher prominence (more distinct extrema)
    if len(peaks_pos) > 0 and len(peaks_neg) > 0:
        avg_prominence_pos = np.mean(properties_pos['prominences']) if len(properties_pos['prominences']) > 0 else 0
        avg_prominence_neg = np.mean(properties_neg['prominences']) if len(properties_neg['prominences']) > 0 else 0

        if avg_prominence_pos >= avg_prominence_neg:
            action_boundaries = peaks_pos
            boundary_type = "maxima"
        else:
            action_boundaries = peaks_neg
            boundary_type = "minima"
    elif len(peaks_pos) > 0:
        action_boundaries = peaks_pos
        boundary_type = "maxima"
    elif len(peaks_neg) > 0:
        action_boundaries = peaks_neg
        boundary_type = "minima"
    else:
        # No peaks found - return single segment spanning all frames
        print("Warning: No cyclic motion detected. Returning single segment.")
        return [[int(frame_indices[0]), int(frame_indices[-1])]]

    print(f"Found {len(action_boundaries)} action boundaries at {boundary_type}")

    # =====================
    # 4. Create start/end times for each action
    # =====================
    # Each action starts at one boundary and ends at the next
    start_and_end_action_times = []

    for i in range(len(action_boundaries) - 1):
        start_idx = action_boundaries[i]
        end_idx = action_boundaries[i + 1]

        # Convert from array index to actual frame index
        start_frame = int(frame_indices[start_idx])
        end_frame = int(frame_indices[end_idx])

        start_and_end_action_times.append([start_frame, end_frame])

    # =====================
    # 5. Plot segmentation results
    # =====================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot motion signal with detected boundaries
    axes[0].plot(frame_indices, motion_signal, 'b-', linewidth=0.8, label='Motion signal (PC1)' if apply_pca_transform else 'Left wrist X')
    axes[0].scatter(frame_indices[action_boundaries], motion_signal[action_boundaries],
                    c='red', s=50, zorder=5, label=f'Action boundaries ({boundary_type})')
    axes[0].set_xlabel('Frame Index')
    axes[0].set_ylabel('Motion Signal')
    axes[0].set_title('Cyclic Motion Detection - Left Wrist')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot segmented actions with alternating colors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (start, end) in enumerate(start_and_end_action_times):
        # Find indices in our arrays
        start_array_idx = np.searchsorted(frame_indices, start)
        end_array_idx = np.searchsorted(frame_indices, end)

        color = colors[i % len(colors)]
        axes[1].axvspan(start, end, alpha=0.3, color=color)
        axes[1].plot(frame_indices[start_array_idx:end_array_idx+1],
                     motion_signal[start_array_idx:end_array_idx+1],
                     color=color, linewidth=1.5, label=f'Action {i+1}: [{start}, {end}]')

    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Motion Signal')
    axes[1].set_title(f'Segmented Actions (n={len(start_and_end_action_times)})')
    if len(start_and_end_action_times) <= 10:
        axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is not None:
        segmentation_path = os.path.join(output_dir, 'action_segmentation.png')
        plt.savefig(segmentation_path, dpi=150)
        print(f"Saved action segmentation plot to: {segmentation_path}")
    plt.close()

    print(f"Detected {len(start_and_end_action_times)} action segments:")
    for i, (start, end) in enumerate(start_and_end_action_times):
        print(f"  Action {i+1}: frames {start} to {end} ({end - start} frames)")

    return start_and_end_action_times


def main(
    results_pkl_path: str,
    video_path: str = "",
    has_world_coords=False,
    is_og_pkl_file=False,
    run_viser=True,
    start_frame = 0,
    end_frame = 176,
    out_folder="/Users/ilonademler/Documents/research/human_motion/caltech-tennis-benchmark/prompthmr/robo/robo_results/robo_formatted_outputs",
    action_title="tennis_forehand",
    timestamps_path: str | None = None,
    timestamp_start: int | None = None,
    timestamp_end: int | None = None,
    save_stick_video: bool = False,
):

    results_dir = Path(results_pkl_path)  # update to your folder
    results_folder = results_dir.stem
    out_folder = Path(out_folder)  # Convert string to Path

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    smplx = SMPLX_Layer(SMPLX_PATH).to(device)

    results_out_folder = out_folder / results_folder


    if (not has_world_coords) and (not is_og_pkl_file):
        world4d = joblib.load(results_pkl_path)
        world4d = {k: world4d[k] for k in range(start_frame, end_frame)}

        all_verts = []
        i = 0

        for k in world4d:
            world3d = world4d[k]
            if len(world3d["track_id"]) == 0:  # no people
                continue

            rotmat = axis_angle_to_matrix(world3d["pose"].reshape(-1, 55, 3))
            global_orient = rotmat[:, :1].to(device)
            body_pose = rotmat[:, 1:22].to(device)
            betas = world3d["shape"].to(device)
            transl = world3d["trans"].to(device) # 1, 3

            # add mujoco transform here
            # prompthmr OG ouptut has z up --> we want y up
            # R_world_to_mj = torch.tensor([
            #     [1, 0,  0],
            #     [0, 0, -1],
            #     [0, 1, 0],
            # ], dtype=torch.float32, device=device)
            R_world_to_mj = torch.tensor([
                [0, 0,  1],
                [1, 0, 0],
                [0, 1, 0],
            ], dtype=torch.float32, device=device)

            global_orient = torch.einsum('ij,njk->nik', R_world_to_mj, global_orient[:, 0].to(device)).unsqueeze(1)
            transl = torch.einsum('ij,tj->ti', R_world_to_mj, transl.to(device))
            rotmat[:,:1] = global_orient
            world3d["pose"] = matrix_to_axis_angle(rotmat)

            if i == 0:
                transl_orig = transl.clone()
                camera_trans_orig = world3d["camera"][:3, 3].copy()
                i += 1


            transl = transl - transl_orig  # apply offset to reduce numerical issues
            world3d['trans'] = transl.cpu()
            world3d['joints'] = world3d['joints'] - transl_orig[:, None, :].cpu()  # apply same offset to joints

            world3d['camera'][:3, 3] -= camera_trans_orig  # c2w: directly offset camera world position

            # rotate camera:
            world3d['camera'][:3, :3] = R_world_to_mj.cpu() @ world3d['camera'][:3, :3]
            world3d["camera"][:3, 3] = R_world_to_mj.cpu() @ world3d["camera"][:3, 3]

            world3d['camera'][:3, 3] = world3d['camera'][:3, 3] + [2, 0, 0]

            pdb.set_trace()
            verts = (
                smplx(
                    global_orient=global_orient.to(device),
                    body_pose=body_pose.to(device),
                    betas=betas.to(device),
                    transl=transl.to(device),
                )
                .vertices.cpu()
                .numpy()
            )
            world3d["vertices"] = verts

            for pid in range(verts.shape[0]):

                min_z = verts[pid, :, 2].min()
                print("min y: ", min_z)
                verts[pid, :, 2] -= min_z  # apply offset here to reduce numerical issues
                world3d['trans'][pid, 2] -= min_z

            all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))

        all_verts = torch.cat(all_verts)  # S, 10475, 3

    elif is_og_pkl_file:
        results = joblib.load(results_pkl_path)

        from fuse_results_robo import create_world4d_w_joints
        world4d = create_world4d_w_joints(results, smplx, device=device)
        world4d = {k: world4d[k] for k in range(start_frame, end_frame)}

        all_verts = []
        i = 0

        floor_z_offset = None
        for k in world4d:
            world3d = world4d[k]
            if len(world3d["track_id"]) == 0:  # no people
                continue

            rotmat = axis_angle_to_matrix(world3d["pose"].reshape(-1, 55, 3))
            if len(rotmat) > 1:
                print("Warning: multiple people detected in frame, only visualizing the first one")
                rotmat = rotmat[:1]
            global_orient = rotmat[:, :1].to(device)
            body_pose = rotmat[:, 1:22].to(device)
            betas = world3d["shape"].to(device)
            transl = world3d["trans"].to(device) # 1, 3

            if betas.shape[0] > 1:
                print("Warning: multiple people detected in frame, only visualizing the first one")
                betas = betas[:1]
            if transl.shape[0] > 1:
                print("Warning: multiple people detected in frame, only visualizing the first one")
                transl = transl[:1]

            # add mujoco transform here
            # prompthmr OG ouptut has z up --> we want y up
            # R_world_to_mj = torch.tensor([
            #     [1, 0,  0],
            #     [0, 0, -1],
            #     [0, 1, 0],
            # ], dtype=torch.float32, device=device)
            R_world_to_mj = torch.tensor([
                [0, 0,  1],
                [1, 0, 0],
                [0, 1, 0],
            ], dtype=torch.float32, device=device)

            global_orient = torch.einsum('ij,njk->nik', R_world_to_mj, global_orient[:, 0].to(device)).unsqueeze(1)
            transl = torch.einsum('ij,tj->ti', R_world_to_mj, transl.to(device))
            rotmat[:,:1] = global_orient
            world3d["pose"] = matrix_to_axis_angle(rotmat)

            if i == 0:
                transl_orig = transl.clone()
                camera_trans_orig = world3d["camera"][:3, 3].copy()
                i += 1


            transl = transl - transl_orig  # apply offset to reduce numerical issues
            world3d['trans'] = transl.cpu()
            world3d['joints'] = world3d['joints'] - transl_orig[:, None, :].cpu()  # apply same offset to joints

            world3d['camera'][:3, 3] -= camera_trans_orig  # c2w: directly offset camera world position

            # rotate camera:
            world3d['camera'][:3, :3] = R_world_to_mj.cpu() @ world3d['camera'][:3, :3]
            world3d["camera"][:3, 3] = R_world_to_mj.cpu() @ world3d["camera"][:3, 3]

            world3d['camera'][:3, 3] = world3d['camera'][:3, 3] + [2, 0, 0]

            # pdb.set_trace()
            verts = (
                smplx(
                    global_orient=global_orient.to(device),
                    body_pose=body_pose.to(device),
                    betas=betas.to(device),
                    transl=transl.to(device),
                )
                .vertices.cpu()
                .numpy()
            )
            world3d["vertices"] = verts

            for pid in range(verts.shape[0]):
                if floor_z_offset is None:
                    floor_z_offset = verts[pid, :, 2].min()

                verts[pid, :, 2] -= floor_z_offset          # visualization: constant offset
                world3d['trans'][pid, 1] -= floor_z_offset   # .npz: constant offset

            all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))

        all_verts = torch.cat(all_verts)  # S, 10475, 3

    else:
        world4d = joblib.load(results_pkl_path)
        world4d = {k: world4d[k] for k in range(start_frame, end_frame)}

        # TODO - shift everything down so that it is level with the ground:
        all_verts = []

        i = 0
        for k in world4d:
            world3d = world4d[k]
            if len(world3d["track_id"]) == 0:  # no people
                continue

            rotmat = axis_angle_to_matrix(world3d["pose_world"].reshape(-1, 55, 3))
            global_orient = rotmat[:, :1].to(device)
            body_pose = rotmat[:, 1:22].to(device)
            betas = world3d["shape"].to(device)
            transl = world3d["trans_world"].to(device) # 1, 3

            # add mujoco transform here
            R_world_to_mj = torch.tensor([
                [-1, 0,  0],
                [0, -1, 0],
                [0, 0,  1],
            ], dtype=torch.float32, device=device)

            global_orient = torch.einsum('ij,njk->nik', R_world_to_mj, global_orient[:, 0].to(device)).unsqueeze(1)
            transl = torch.einsum('ij,tj->ti', R_world_to_mj, transl.to(device))

            if i == 0:
                transl_orig = transl.clone()
                camera_trans_orig = world3d["camera"][:3, 3].copy()
                i += 1


            transl = transl - transl_orig  # apply offset to reduce numerical issues
            world3d['trans_world'] = transl.cpu()
            world3d['joints_world'] = world3d['joints_world'] - transl_orig[:, None, :].cpu()  # apply same offset to joints

            print("trans before: ", world3d["camera"][:3, 3])
            world3d['camera'][:3, 3] -= camera_trans_orig  # c2w: directly offset camera world position
            print("trans after: ", world3d["camera"][:3, 3])

            # rotate camera:
            world3d['camera'][:3, :3] = R_world_to_mj.cpu() @ world3d['camera'][:3, :3]
            world3d["camera"][:3, 3] = R_world_to_mj.cpu() @ world3d["camera"][:3, 3]

            world3d['camera'][:3, 3] = world3d['camera'][:3, 3] + [2, 0, 0]

            verts = (
                smplx(
                    global_orient=global_orient.to(device),
                    body_pose=body_pose.to(device),
                    betas=betas.to(device),
                    transl=transl.to(device),
                )
                .vertices.cpu()
                .numpy()
            )
            world3d["vertices"] = verts

            for pid in range(verts.shape[0]):

                left_big_toe = world3d['joints_world'][pid,19][2].numpy()
                right_big_toe = world3d['joints_world'][pid,22][2].numpy()
                left_pinky_toe = world3d['joints_world'][pid,20][2].numpy()
                right_pinky_toe = world3d['joints_world'][pid,23][2].numpy()
                left_heel = world3d['joints_world'][pid,21][2].numpy()
                right_heel = world3d['joints_world'][pid,24][2].numpy()

                min_z = min(
                    left_big_toe,
                    right_big_toe,
                    left_pinky_toe,
                    right_pinky_toe,
                    left_heel,
                    right_heel,
                    )
                print("min foot height: ", min_z)
                verts[pid, :, 2] -= min_z  # apply offset here to reduce numerical issues
                world3d['trans_world'][pid, 2] -= min_z

                # for visualization purposes overwrite the prompthmr relative coords
                world3d["trans"] = world3d["trans_world"]
                world3d["pose"] = world3d["pose_world"]

            all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))

        all_verts = torch.cat(all_verts)  # S, 10475, 3

    smplx_pose_body = np.array([world4d[i]['pose'].reshape(-1,165)[0, 3:66] for i in range(start_frame, end_frame)])  # T x 63
    smplx_beta = np.array([world4d[i]['shape'][0,:] for i in range(start_frame, end_frame)]) # T x 10
    smplx_root_orient = np.array([world4d[i]['pose'].reshape(-1,165)[0, :3] for i in range(start_frame, end_frame)])   # T x 3
    smplx_trans = np.array([world4d[i]['trans'][0,:] for i in range(start_frame, end_frame)]) # T x 3

    smplx_root_orient = np.array(smplx_root_orient)
    smplx_trans = np.array(smplx_trans)

    # pad smplx_beta with 0s:
    T = smplx_beta.shape[0]
    pad = np.zeros((T, 6), dtype=smplx_beta.dtype)
    smplx_beta_16 = np.concatenate([smplx_beta, pad], axis=1)
    smplx_beta_16 = smplx_beta_16[0:1,:]

    print(smplx_pose_body.shape)
    print(smplx_beta_16.shape)
    print(smplx_root_orient.shape)
    print(smplx_trans.shape)

    retargeting_file = {
    "pose_body": smplx_pose_body, # T, 63
    "betas": smplx_beta_16, # 1, 16
    "root_orient": smplx_root_orient, # T, 55, 3, 3
    "trans": smplx_trans, # T, 3
    "gender": "neutral",
    "mocap_frame_rate": 30,
    }

    print("ll_verts[:,:,2].min(): ", all_verts[:,:,2].min())


    out_path = out_folder / f"{action_title}.npz"

    os.makedirs(out_folder, exist_ok=True)
    np.savez(out_path, **retargeting_file)
    print("Saved retargeting file to:", out_path)

    if save_stick_video:
        stick_video_path = out_folder / f"{action_title}_stick_figure.mp4"
        save_stick_figure_video(world4d, stick_video_path, fps=30)

    if timestamps_path is not None:
        start_frame_idx = int(results_pkl_path.split("/")[-1].split(".")[0].split("_")[-2]) # index 0 of video crop timestamp
        timestamps = np.load(timestamps_path)
        print("Loaded timestamps from:", timestamps_path)

        timestamp_start_idx = timestamp_start + start_frame_idx # timestamp of action start
        timestamp_end_idx = timestamp_end + start_frame_idx # timestamp of action end

        video_start_time = timestamps[start_frame_idx + start_frame] # timestamp of video start
        action_start_time = timestamps[timestamp_start_idx] - video_start_time # relative start of action
        action_end_time = timestamps[timestamp_end_idx] - video_start_time # relative end of action

        print(f"\n\nSelected timestamp range: {action_start_time} ms to {action_end_time} ms\n\n")

        pdb.set_trace()

    if run_viser:
        floor_verts, floor_faces, _ = get_floor_mesh_z_up(all_verts, z_offset=0.0)
        floor = (floor_verts, floor_faces.astype(np.int32))

        if video_path:
            # Load video frames for the selected range
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            images = []
            for _ in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            print(f"Loaded {len(images)} video frames from {video_path}")

            server, gui = viser_vis_world4d_z_up(
                images,
                world4d,
                faces=smplx.faces,
                floor=floor,
                block=True,
            )
        else:
            pdb.set_trace()
            server, gui = viser_vis_world4d_no_video_z_up(
                world4d,
                faces=smplx.faces,
                floor=floor,
                block=True,
            )

        url = f"https://localhost:{server.get_port()}"
        print(f"Please use this url to view the results: {url}")
        print(
            "For longer video, it will take a few seconds for the webpage to load."
        )

        gui_playing, gui_timestep, gui_framerate, num_frames = gui
        while True:
            # Update the timestep if we're playing.
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames

            time.sleep(1.0 / gui_framerate.value)




if __name__ == "__main__":
    print("hi")
    tyro.cli(main)
