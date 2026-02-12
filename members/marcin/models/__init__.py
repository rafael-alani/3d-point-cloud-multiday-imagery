"""Inpainting engines for satellite image restoration."""

from .deep_image_prior import DIPEngine, DIP_RESTORATION_DEFAULTS, DIP_STITCHING_DEFAULTS
from .satdiff import SatDiffEngine, RESTORATION_DEFAULTS, STITCHING_DEFAULTS
from .utils import postprocess


# Registry of available engines
AVAILABLE_ENGINES = {
    "Deep Image Prior": DIPEngine,
    "Diffusion": SatDiffEngine,
}
