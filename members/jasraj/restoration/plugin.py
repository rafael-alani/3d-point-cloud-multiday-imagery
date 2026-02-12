from interface import SatellitePlugin
import numpy as np
from .core import RestorationEngine

class RestorationPlugin(SatellitePlugin):
    """
    Image restoration pipeline for satellite imagery:
    - Non-Local Means Denoising
    - Dark Channel Prior Dehazing
    - Unsharp Mask Sharpening
    """
    
    def __init__(self):
        self.engine = RestorationEngine()
    
    @property
    def name(self):
        return "Image Restoration (Denoise + Dehaze)"
    
    def run(self, image: np.ndarray):
        """
        Run full restoration pipeline on input image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            List of napari layers showing restoration stages:
            - Original (degraded) input
            - Denoised result
            - Dehazed result
            - Final sharpened result
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Handle RGB extraction if needed
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        
        print("Running Restoration Pipeline...")
        
        # Stage 1: Denoising
        print("  1/3 Denoising (NLM)...")
        denoised = self.engine.run_denoising_nlm(image)
        
        # Stage 2: Dehazing
        print("  2/3 Dehazing (Dark Channel Prior)...")
        dehazed, transmission_map = self.engine.run_dehazing_dcp(image)
        
        # Stage 3: Sharpening
        print("  3/3 Sharpening...")
        final_result = self.engine.run_sharpening(dehazed)
        
        print("Restoration complete!")
        
        return [
            (image, {
                "name": "[Restoration] Input",
                "rgb": True,
                "opacity": 1.0,
                "visible": False
            }, "image"),
            (denoised, {
                "name": "[Restoration] Denoised",
                "rgb": True,
                "opacity": 1.0,
                "visible": False
            }, "image"),
            (dehazed, {
                "name": "[Restoration] Dehazed",
                "rgb": True,
                "opacity": 1.0,
                "visible": False
            }, "image"),
            (final_result, {
                "name": "[Restoration] Output",
                "rgb": True,
                "opacity": 1.0,
                "visible": True
            }, "image"),
            (transmission_map, {
                "name": "[Restoration] Transmission Map",
                "opacity": 0.7,
                "colormap": "viridis",
                "visible": False
            }, "image")
        ]
