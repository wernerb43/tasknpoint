import os

PROMPTHMR_DATA_ROOT = os.environ.get("PROMPTHMR_DATA_ROOT", "")
PROMPTHMR_PRETRAIN_ROOT = os.environ.get("PROMPTHMR_PRETRAIN_ROOT", PROMPTHMR_DATA_ROOT)

PRETRAIN_DIR = f"{PROMPTHMR_PRETRAIN_ROOT}/pretrain"

SMPLX_PATH = f"{PROMPTHMR_DATA_ROOT}/body_models/smplx"
SMPL_PATH = f"{PROMPTHMR_DATA_ROOT}/body_models/smpl"

SMPLX_NEUTRAL_MODEL = f"{SMPLX_PATH}/SMPLX_NEUTRAL.npz"
SMPLX_NEUTRAL_ARRAY_MODEL = f"{SMPLX_PATH}/SMPLX_neutral_array_f32_slim.npz"

CHECKPOINTS = {
    "yolo": f"{PRETRAIN_DIR}/yolo11x.pt",
    "sam": f"{PRETRAIN_DIR}/sam_vit_h_4b8939.pth",
    "sam2": f"{PRETRAIN_DIR}/sam2_ckpts",
    "vitpose": f"{PRETRAIN_DIR}/vitpose-h-coco_25.pth",
    "droid": f"{PRETRAIN_DIR}/droid.pth",
    "droidcalib": f"{PRETRAIN_DIR}/droidcalib.pth",
    "camcalib": f"{PRETRAIN_DIR}/camcalib_sa_biased_l2.ckpt",
    "phmr": f"{PRETRAIN_DIR}/phmr/checkpoint.ckpt",
    "phmr_vid_config": f"{PRETRAIN_DIR}/phmr_vid/prhmr_release_002.yaml",
    "phmr_vid_ckpt": f"{PRETRAIN_DIR}/phmr_vid/prhmr_release_002.ckpt",
}
