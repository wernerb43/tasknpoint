import os
import sys

# Resolve `from prompthmr.pipeline import Pipeline` without pip-installing caltennis
_phmr = os.environ.get("PHMR_REPO", "")
if _phmr and _phmr not in sys.path:
    sys.path.insert(0, _phmr)

import torch
import tyro
from typing import Optional

from prompthmr.pipeline import Pipeline
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from utils import get_smplx_path, estimate_num_frames


def main(
    input_video: str,
    output_dir: str = "results",
    static_camera: bool = True,
    run_viser: bool = False,
    viser_total: int = 3000,
    viser_subsample: int = 1,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    chunk_size: int = 500,
):
    SMPLX_PATH = get_smplx_path()
    print("SMPLX_PATH:", SMPLX_PATH)
    smplx = SMPLX_Layer(SMPLX_PATH).cuda()

    session_name = input_video.split("/")[-2]
    output_folder = f"{output_dir}/{session_name}/" + os.path.basename(input_video).split(".")[0]
    pipeline = Pipeline(static_cam=static_camera)

    num_frames = estimate_num_frames(input_video)
    print(f"Estimated frames: {num_frames}")

    if num_frames > chunk_size and end_frame is None:
        for i in range(0, num_frames, chunk_size // 2):
            ef = min(i + chunk_size, num_frames)
            print(f"Processing frames {i} to {ef} of {num_frames}")
            try:
                results = pipeline(
                    input_video, output_folder, save_only_essential=True,
                    start_frame=i, end_frame=ef - 1, save_joint_info=True,
                )
                pipeline.images = None
                pipeline.results = None
                torch.cuda.empty_cache()
            except ValueError as e:
                if "No persons detected" in str(e):
                    print(f"Skipping frames {i}-{ef} (no persons).")
                else:
                    raise
    else:
        sf = start_frame or 0
        ef = end_frame if end_frame is not None else num_frames - 1
        pipeline(input_video, output_folder, save_only_essential=True,
                 start_frame=sf, end_frame=ef, save_joint_info=True)


if __name__ == "__main__":
    tyro.cli(main)
