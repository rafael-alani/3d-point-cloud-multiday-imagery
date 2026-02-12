"""Image restoration processing pipeline."""

import cv2
import numpy as np

from ..models import AVAILABLE_ENGINES, postprocess


def prefill_telea(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked regions using OpenCV Telea inpainting as initialization."""
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    mask_uint8 = mask.astype(np.uint8)

    result = cv2.inpaint(image_uint8, mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return result.astype(np.float32) / 255.0


def create_nan_mask(nan_mask: np.ndarray, margin: int = 0) -> np.ndarray:
    """Convert boolean NaN mask to uint8 with optional dilation margin."""
    mask = (nan_mask * 255).astype(np.uint8)

    if margin > 0:
        kernel_size = margin * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel)

    return mask
    


class ImageRestorationProcessor:
    """Orchestrates the restoration pipeline using selected engine."""

    def __init__(self, engine_name: str = "Diffusion"):
        engine_class = AVAILABLE_ENGINES.get(engine_name)
        if engine_class is None:
            raise ValueError(f"Unknown engine: {engine_name}")
        self.engine = engine_class()

    def process(
        self,
        image: np.ndarray,
        nan_mask: np.ndarray,
        margin: int = 10,
        **config,
    ):
        """Run full restoration pipeline on image with NaN regions."""
        mask = create_nan_mask(nan_mask, margin)

        # Nothing to restore
        if mask.sum() == 0:
            return image.copy(), image.copy()

        # Log restoration scope
        masked_pixels = (mask > 0).sum()
        masked_percent = 100 * (mask > 0).mean()
        print(f"Restoring {masked_pixels:,} px ({masked_percent:.1f}%)")

        working_image = prefill_telea(image, mask)

        # Run engine and blend result
        engine_output = self.engine.restore(working_image, mask, **config)
        result = postprocess(working_image, engine_output, mask)

        return image.copy(), result
