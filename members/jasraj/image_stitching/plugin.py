from interface import SatellitePlugin
import numpy as np
from .core import ImageStitcher

class ImageStitchingPlugin(SatellitePlugin):
    """
    Advanced image stitching using SIFT features with spatial consistency
    matching and optimal seam finding via dynamic programming.

    Select two overlapping images, then click "Run" to stitch them together.
    """

    def __init__(self):
        self.stitcher = ImageStitcher(verbose=True)

    @property
    def name(self):
        return "Image Stitching (SIFT + Seam Cut)"

    def run(self, image: np.ndarray, image2: np.ndarray = None):
        """
        Stitch two overlapping images together.

        Args:
            image: First RGB image as numpy array (H, W, 3)
            image2: Second RGB image as numpy array (H, W, 3)

        Returns:
            List of napari layers with the stitched result
        """
        if image2 is None:
            return [
                (image, {
                    "name": "[Stitching] Error: Second image not provided",
                    "rgb": True,
                    "opacity": 1.0
                }, "image")
            ]

        img1 = image
        img2 = image2

        print(f"\n🔧 Stitching Images:")
        print(f"   Image 1: {img1.shape}")
        print(f"   Image 2: {img2.shape}")
        
        # Convert to uint8 RGB if needed
        img1 = self._prepare_image(img1)
        img2 = self._prepare_image(img2)
        
        # Run stitching
        try:
            result = self.stitcher.stitch(img1, img2)
            
            if result is None:
                return [
                    (image, {
                        "name": "[Stitching] Failed - check console",
                        "rgb": True,
                        "opacity": 1.0
                    }, "image")
                ]
            
            print(f"✅ Stitching complete! Result shape: {result.shape}")

            return [
                (img1, {
                    "name": "[Stitching] Input 1",
                    "rgb": True,
                    "opacity": 1.0,
                    "visible": False
                }, "image"),
                (img2, {
                    "name": "[Stitching] Input 2",
                    "rgb": True,
                    "opacity": 1.0,
                    "visible": False
                }, "image"),
                (result, {
                    "name": "[Stitching] Output",
                    "rgb": True,
                    "opacity": 1.0
                }, "image")
            ]
            
        except Exception as e:
            print(f"❌ Stitching error: {e}")
            import traceback
            traceback.print_exc()
            return [
                (image, {
                    "name": f"[Stitching] Error: {str(e)[:30]}",
                    "rgb": True,
                    "opacity": 1.0
                }, "image")
            ]
    
    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        """
        Convert image to uint8 RGB format.
        
        Args:
            img: Input image (any format)
            
        Returns:
            RGB uint8 image (H, W, 3)
        """
        # Handle grayscale
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        
        # Handle RGBA -> RGB
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Handle multichannel (take first 3)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        
        # Convert to uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        return img
