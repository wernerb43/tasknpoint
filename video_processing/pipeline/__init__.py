from pkgutil import extend_path
# Extend __path__ so all unchanged submodules (detector, camera, gvhmr, etc.)
# are resolved from the PromptHMR repo already on sys.path.
__path__ = extend_path(__path__, __name__)

from .pipeline import Pipeline
