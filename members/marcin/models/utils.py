"""Utility functions for model outputs."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def postprocess(original: np.ndarray, result: np.ndarray, mask: np.ndarray,
                blend_width: int = 5) -> np.ndarray:
    """Blend restored regions smoothly into the original image."""
    mask_bool = mask == 255

    if not mask_bool.any():
        return original.copy()

    # Distance from mask edge for smooth blending
    distance = distance_transform_edt(mask_bool)
    blend_factor = np.clip(distance / max(blend_width, 1), 0, 1)

    # Only blend inside the mask
    blend_factor = np.where(mask_bool, blend_factor, 0)[:, :, None]

    return np.clip(original * (1 - blend_factor) + result * blend_factor, 0, 1)
