from interface import SatellitePlugin
import numpy as np
from .core import OBIAClassifier

class LandUseClassificationPlugin(SatellitePlugin):
    """
    OBIA-based land use classifier using SLIC superpixels, 
    Gabor texture features, and local entropy analysis.
    """
    
    def __init__(self, n_segments=1500, n_clusters=5):
        """
        Args:
            n_segments: Number of SLIC superpixels (default: 1500)
            n_clusters: Number of K-means clusters for classification (default: 5)
        """
        self.n_segments = n_segments
        self.n_clusters = n_clusters
        self.classifier = OBIAClassifier()
    
    @property
    def name(self):
        return "Land Use Classification (OBIA)"
    
    def run(self, image: np.ndarray):
        """
        Run OBIA classification pipeline on the input image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            List of napari layers:
            - Classification map (labels layer)
            - Superpixel boundaries (image layer)
            - PCA feature visualization (image layer)
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Handle RGB extraction if needed
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        
        print(f"Running OBIA Classification with {self.n_segments} segments, {self.n_clusters} clusters...")
        
        # Run the OBIA pipeline
        class_map, boundaries, pca_vis = self.classifier.run_obia_pipeline(
            image, 
            n_segments=self.n_segments, 
            n_clusters=self.n_clusters
        )
        
        return [
            (image, {
                "name": "[Classification] Input",
                "rgb": True,
                "opacity": 1.0,
                "visible": False
            }, "image"),
            (boundaries, {
                "name": "[Classification] Superpixels",
                "rgb": True,
                "opacity": 0.5,
                "visible": False
            }, "image"),
            (pca_vis, {
                "name": "[Classification] Features (PCA)",
                "rgb": True,
                "opacity": 0.6,
                "visible": False
            }, "image"),
            (class_map, {
                "name": f"[Classification] Output (K={self.n_clusters})",
                "opacity": 0.7
            }, "labels"),
        ]
