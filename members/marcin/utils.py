"""Image loading utilities for Sentinel-2 satellite data."""

import numpy as np
import tifffile


def normalize_band(band, preserve_nan=False):
    """Normalize band to 0-1 range using 2-98 percentile stretch."""

    if preserve_nan:
        result = band.copy().astype(np.float32)
        valid_mask = ~np.isnan(band) & (band > 0)

        if valid_mask.sum() == 0:
            return np.nan_to_num(result, nan=0.0)

        low, high = np.percentile(band[valid_mask], [2, 98])

        if high != low:
            normalized = (band[valid_mask] - low) / (high - low)
            result[valid_mask] = np.clip(normalized, 0, 1)
        else:
            result[valid_mask] = 0

        return result

    # Standard case: replace NaN with 0 first
    band = np.nan_to_num(band, nan=0.0)
    valid_pixels = band[band > 0]

    if len(valid_pixels) == 0:
        return np.zeros_like(band)

    low, high = np.percentile(valid_pixels, [2, 98])

    if high == low:
        return np.zeros_like(band)

    normalized = (band - low) / (high - low)
    return np.clip(normalized, 0, 1)


def load_image_as_rgb(path):
    """Load TIF file and extract RGB composite (bands 4, 3, 2)."""

    data = tifffile.imread(path)

    # Sentinel-2: B04=Red, B03=Green, B02=Blue (indices 3, 2, 1)
    red = normalize_band(data[:, :, 3])
    green = normalize_band(data[:, :, 2])
    blue = normalize_band(data[:, :, 1])

    rgb = np.stack([red, green, blue], axis=-1)
    return rgb.astype(np.float32)


def load_image_with_nans(path):
    """Load TIF file and return RGB image with NaN mask."""

    data = tifffile.imread(path)

    # Find NaN pixels in any of the RGB bands
    nan_mask = (
        np.isnan(data[:, :, 3]) |
        np.isnan(data[:, :, 2]) |
        np.isnan(data[:, :, 1])
    )

    # Normalize each band, preserving NaN locations
    red = normalize_band(data[:, :, 3], preserve_nan=True)
    green = normalize_band(data[:, :, 2], preserve_nan=True)
    blue = normalize_band(data[:, :, 1], preserve_nan=True)

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)

    return rgb.astype(np.float32), nan_mask
