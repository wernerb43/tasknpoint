import os
import subprocess


def get_smplx_path():
    return os.path.join(os.environ.get("PROMPTHMR_DATA_ROOT", ""), "body_models", "smplx")


def estimate_num_frames(video_path):
    """Count video frames via ffprobe (ported from caltennis video_alignment_utils)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return int(result.stdout)
