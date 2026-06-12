from pkgutil import extend_path
import os as _os, sys as _sys

# extend_path finds PromptHMR's pipeline/ when installed with legacy editable mode
# (pip install -e . --config-settings editable_mode=compat).
__path__ = extend_path(__path__, __name__)

# Always locate the PromptHMR root via prompt_hmr — works with any install mode.
try:
    import prompt_hmr as _phmr
    _phmr_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(_phmr.__file__)))
    del _phmr
except ImportError:
    raise ImportError(
        "prompt_hmr is not importable. Install PromptHMR in this environment:\n"
        "  cd /path/to/PromptHMR && pip install -e ."
    )

# New-style editable installs (PEP 660) don't add the repo root as a plain sys.path entry,
# so extend_path misses PromptHMR's pipeline/. Add it explicitly.
_phmr_pipeline = _os.path.join(_phmr_root, "pipeline")
if _os.path.isdir(_phmr_pipeline) and _phmr_pipeline not in __path__:
    __path__.append(_phmr_pipeline)
if _phmr_root not in _sys.path:
    _sys.path.insert(0, _phmr_root)
del _phmr_pipeline

# droid.py lives in pipeline/droidcalib/droid_slam/ inside the PromptHMR repo.
_phmr_droid = _os.path.join(_phmr_root, "pipeline", "droidcalib", "droid_slam")
if _os.path.isdir(_phmr_droid) and _phmr_droid not in _sys.path:
    _sys.path.insert(0, _phmr_droid)
del _phmr_droid, _phmr_root, _os, _sys

from .pipeline import Pipeline
