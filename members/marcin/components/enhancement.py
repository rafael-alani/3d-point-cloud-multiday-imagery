"""Detail enhancement processing pipeline."""

import cv2
import numpy as np

from ..models import AVAILABLE_ENGINES, postprocess


def blend_high_frequency(
    original: np.ndarray, model_output: np.ndarray, hf_blend: float
) -> np.ndarray:
    """Blend high-frequency details from original into model output.
    hf_blend: Blend factor (0 = no blend, 1 = full detail restoration)

    """
    if hf_blend <= 0:
        return model_output

    # Extract high-frequency using Gaussian blur
    sigma = 1.5
    ksize = int(sigma * 4) | 1  # Ensure odd kernel size

    blurred = cv2.GaussianBlur(original, (ksize, ksize), sigma)
    high_freq = original - blurred

    # Add weighted high-frequency to model output
    result = model_output + hf_blend * high_freq
    result = np.clip(result, 0.0, 1.0)

    return result.astype(np.float32)


class EnhancementProcessor:
    """Processor for detail enhancement using inpainting models."""

    def __init__(self, engine_name: str = "Diffusion"):
        engine_class = AVAILABLE_ENGINES.get(engine_name)
        if engine_class is None:
            raise ValueError(f"Unknown engine: {engine_name}")
        self.engine = engine_class()

    def process(
        self,
        image: np.ndarray,
        hf_blend: float = 0.5,
        **config,
    ):
        """Run enhancement on the image."""
        h, w = image.shape[:2]

        print(f"Enhancing {w}x{h} image...")

        # Create full mask (enhance entire image)
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Run engine enhancement
        print(f"Running enhancement...")
        engine_output = self.engine.enhance(image, mask, **config)

        # Apply HF blend to restore sharpness
        result = blend_high_frequency(image, engine_output, hf_blend)

        # Postprocess for edge blending (minimal effect with full mask)
        result = postprocess(image, result, mask, blend_width=0)

        return image.copy(), result
