from interface import SatellitePlugin
import numpy as np
from .saliency import spectral_residual_global_detection

PREFIX = "[Saliency]"


def _normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalize a single band to 0-1 using 2-98 percentile."""
    band = np.nan_to_num(band, nan=0.0)
    valid = band > 0
    if not valid.any():
        return np.zeros_like(band, dtype=np.float32)
    low, high = np.percentile(band[valid], [2, 98])
    if high == low:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - low) / (high - low), 0, 1).astype(np.float32)


def _prepare_input_rgb(image: np.ndarray) -> np.ndarray:
    """Convert multi-band image to RGB for display."""
    # Handle channel-first format (C, H, W) -> (H, W, C)
    if image.ndim == 3 and image.shape[0] <= 4 and image.shape[1] > 4 and image.shape[2] > 4:
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 3 and image.shape[2] > 3:
        # Multi-channel - use bands 4,3,2 (R,G,B for Sentinel-2)
        r = _normalize_band(image[:, :, 3])
        g = _normalize_band(image[:, :, 2])
        b = _normalize_band(image[:, :, 1])
        return np.stack([r, g, b], axis=-1)
    elif image.ndim == 2:
        gray = _normalize_band(image)
        return np.stack([gray, gray, gray], axis=-1)
    else:
        # Already RGB (H, W, 3)
        img = np.nan_to_num(image, nan=0.0).astype(np.float32)
        if img.max() > 1:
            img = img / 255.0
        return img


class SaliencyDetector(SatellitePlugin):
    @property
    def name(self):
        return "Saliency Object Detection"

    def run(self, image: np.ndarray):
        # Prepare input for display
        input_rgb = _prepare_input_rgb(image)

        _, saliency_map, boxes = spectral_residual_global_detection(image)

        shapes_data = []
        for (x, y, w, h) in boxes:
            shapes_data.append(np.array([[y, x], [y+h, x+w]]))

        # Normalize saliency map for display
        saliency_display = saliency_map.astype(np.float32)
        if saliency_display.max() > 1:
            saliency_display = saliency_display / 255.0

        return [
            (input_rgb, {"name": f"{PREFIX} Input", "rgb": True}, "image"),
            (saliency_display, {"name": f"{PREFIX} Saliency Map", "colormap": "inferno", "opacity": 0.7}, "image"),
            (shapes_data, {
                "shape_type": "rectangle",
                "name": f"{PREFIX} Output",
                "edge_color": "red",
                "face_color": "transparent",
                "edge_width": 4
            }, "shapes")
        ]
