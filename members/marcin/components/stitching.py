import cv2
import numpy as np

from ..models import AVAILABLE_ENGINES, postprocess


def prefill_telea(composite: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pre-fill the gap region with Telea inpainting."""
    mask_uint8 = mask.astype(np.uint8)
    img_uint8 = (composite * 255).astype(np.uint8)
    inpainted = cv2.inpaint(img_uint8, mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted.astype(np.float32) / 255.0


class StitchingProcessor:
    """Processor for stitching multiple satellite image"""

    def __init__(self, engine_name: str = "Diffusion"):
        engine_class = AVAILABLE_ENGINES.get(engine_name)
        if engine_class is None:
            raise ValueError(f"Unknown engine: {engine_name}")
        self.engine = engine_class()

    def process(
        self,
        images: list,
        overlap: int = 128,
        blend_width: int = 5,
        **config,
    ):
        """Run the full stitching pipeline"""
        left, right = images[0], images[1]

        # Create input preview (full images touching, no gap, no clipping)
        input_preview = self._create_input_preview(left, right)

        # Create processing composite (with gap for inpainting)
        composite, mask = self._create_horizontal_composite(left, right, overlap)

        # Pre-fill gap with telea
        composite = prefill_telea(composite, mask)

        # Run engine
        engine_output = self.engine.stitch(composite, mask, **config)
        result = postprocess(composite, engine_output, mask, blend_width=blend_width)

        return input_preview, result

    def _create_input_preview(self, left: np.ndarray, right: np.ndarray):
        """Create preview with full images side by side, touching."""
        h1, w1 = left.shape[:2]
        h2, w2 = right.shape[:2]

        h = min(h1, h2)
        preview = np.zeros((h, w1 + w2, 3), dtype=np.float32)
        preview[:, :w1] = left[:h]
        preview[:, w1:] = right[:h]

        return preview

    def _create_horizontal_composite(
        self, left: np.ndarray, right: np.ndarray, overlap: int
    ):
        """Create horizontal composite with overlap gap."""
        h1, w1 = left.shape[:2]
        h2, w2 = right.shape[:2]

        h = min(h1, h2)
        left = left[:h]
        right = right[:h]

        left_keep = w1 - overlap // 2
        right_keep = w2 - overlap // 2
        total_w = left_keep + overlap + right_keep

        composite = np.zeros((h, total_w, 3), dtype=np.float32)
        composite[:, :left_keep] = left[:, :left_keep]
        composite[:, left_keep + overlap:] = right[:, w2 - right_keep:]

        mask = np.zeros((h, total_w), dtype=np.uint8)
        mask[:, left_keep:left_keep + overlap] = 255

        return composite, mask

