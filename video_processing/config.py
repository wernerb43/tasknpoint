import os
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("config.env")


def _parse_value(raw_value):
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return os.path.expanduser(os.path.expandvars(value))


def _load_env_file(path):
    data = {}
    if not path.is_file():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        data[key.strip()] = _parse_value(raw_value)
    return data


_FILE_CONFIG = _load_env_file(CONFIG_PATH)


def get_config(name, default=None):
    return os.environ.get(name, _FILE_CONFIG.get(name, default))


PROMPTHMR_DATA_ROOT = get_config("PROMPTHMR_DATA_ROOT", "")
PROMPTHMR_PRETRAIN_ROOT = get_config("PROMPTHMR_PRETRAIN_ROOT", PROMPTHMR_DATA_ROOT)
RESULTS_ROOT = get_config("RESULTS_ROOT", "")
RETARGET_OUTPUTS_ROOT = get_config("RETARGET_OUTPUTS_ROOT", "")

SMPLX_PATH = str(Path(PROMPTHMR_DATA_ROOT) / "body_models" / "smplx") if PROMPTHMR_DATA_ROOT else ""
