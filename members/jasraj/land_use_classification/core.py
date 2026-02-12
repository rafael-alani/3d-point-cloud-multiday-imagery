"""
Core functionality for Object-Based Image Analysis (OBIA) land use classification.

This module implements an advanced OBIA classifier using:
- SLIC superpixel segmentation
- Gabor texture features
- Local entropy analysis
- K-means clustering on aggregated object features
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.segmentation import slic, mark_boundaries
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte


# --- Utility Functions ---

def normalize_band(band):
    """
    Normalize a single band to 0-255 uint8 range using percentile stretch.
    
    Args:
        band: Input band as numpy array
        
    Returns:
        Normalized band as uint8
    """
    band = np.nan_to_num(band, nan=0.0)
    valid = band[band > 0]
    if len(valid) == 0:
        return np.zeros_like(band, dtype=np.uint8)
    vmin, vmax = np.percentile(valid, [2, 98])
    return ((np.clip((band - vmin) / (vmax - vmin), 0, 1)) * 255).astype(np.uint8)


def smart_crop(image):
    """
    Automatically crop image to remove zero-valued borders.
    
    Args:
        image: Input image (2D or 3D array)
        
    Returns:
        Cropped image
    """
    if image.ndim == 3:
        mask = np.sum(image, axis=2) > 0
    else:
        mask = image > 0
    
    if not np.any(mask):
        return image
    
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    return image[y_min:y_max, x_min:x_max]


# --- Main Classifier ---

class OBIAClassifier:
    """
    Object-Based Image Analysis classifier for land use classification.
    
    Pipeline:
    1. SLIC superpixel segmentation
    2. Pixel-level feature extraction (LAB color, Gabor texture, entropy)
    3. Feature aggregation per superpixel
    4. K-means clustering on object features
    5. Classification map reconstruction
    """
    
    def __init__(self, verbose=False):
        """
        Initialize the OBIA classifier.
        
        Args:
            verbose: Print progress messages (default: True)
        """
        self.verbose = verbose
        self.filters = self.build_gabor_filters()
        if self.verbose:
            print(f"Initialized Gabor Bank: {len(self.filters)} filters")
    
    def build_gabor_filters(self):
        """
        Build a bank of Gabor filters for texture analysis.
        
        Returns:
            List of Gabor filter kernels
        """
        filters = []
        ksize = 31
        
        # Cover all directions: 0°, 45°, 90°, 135°
        for theta in np.arange(0, np.pi, np.pi / 4):
            for sigma in (3, 5):  # Two scales
                for lambd in (10, 15):  # Two wavelengths
                    kern = cv2.getGaborKernel(
                        (ksize, ksize), sigma, theta, lambd, 0.5, 0, 
                        ktype=cv2.CV_32F
                    )
                    filters.append(kern)
        
        return filters
    
    def get_local_entropy(self, image_gray):
        """
        Calculate local entropy (texture randomness).
        
        High entropy indicates urban areas or forests.
        Low entropy indicates water or sand.
        
        Args:
            image_gray: Grayscale uint8 image
            
        Returns:
            Entropy map
        """
        return entropy(image_gray, disk(5))
    
    def extract_pixel_features(self, image):
        """
        Extract multi-modal features for every pixel.
        
        Features include:
        - LAB color space (3 channels)
        - Local entropy (1 channel)
        - Gabor texture responses (16 channels)
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Feature stack (H, W, D) where D is number of features
        """
        # A. Color Space Transformation (RGB -> LAB)
        # Lab separates Luminance (L) from Color (a,b), reducing shadow impact
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # B. Gabor Texture Features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gabor_feats = []
        
        for kern in self.filters:
            fimg = cv2.filter2D(gray, cv2.CV_8UC3, kern)
            fimg = cv2.GaussianBlur(fimg.astype(np.float32), (15, 15), 0)
            gabor_feats.append(fimg)
        
        gabor_stack = np.stack(gabor_feats, axis=2)
        
        # C. Local Entropy Feature
        gray_u8 = img_as_ubyte(normalize_band(gray))
        ent_map = self.get_local_entropy(gray_u8).astype(np.float32)
        ent_map = np.expand_dims(ent_map, axis=2)
        
        # Stack: [L, a, b, Entropy, Gabor1, ..., Gabor16]
        full_stack = np.concatenate([lab_image, ent_map, gabor_stack], axis=2)
        return full_stack
    
    def run_obia_pipeline(self, image, n_segments=1000, n_clusters=4):
        """
        Run the complete OBIA classification pipeline.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            n_segments: Number of SLIC superpixels (default: 1000)
            n_clusters: Number of K-means clusters (default: 4)
            
        Returns:
            Tuple of (class_map, boundaries, pca_vis):
            - class_map: Classification map (H, W) with cluster labels
            - boundaries: Image with superpixel boundaries overlaid
            - pca_vis: PCA visualization of object features (H, W, 3)
        """
        h, w = image.shape[:2]
        
        # 1. SLIC Superpixel Segmentation
        if self.verbose:
            print(f"1. Segmenting image into ~{n_segments} Superpixels (SLIC)...")
        
        segments = slic(
            image, 
            n_segments=n_segments, 
            compactness=20,  # Makes segments more square
            sigma=1,  # Gaussian smoothing before segmentation
            start_label=1
        )
        n_actual_segments = segments.max()
        
        # 2. Extract Pixel-Level Features
        if self.verbose:
            print("2. Extracting Deep Texture & Color Features...")
        
        pixel_features = self.extract_pixel_features(image)
        n_features = pixel_features.shape[2]
        
        # 3. Feature Aggregation (The OBIA Step)
        # Compute mean feature vector for each superpixel
        if self.verbose:
            print("3. Aggregating Features per Superpixel...")
        
        superpixel_features = np.zeros((n_actual_segments + 1, n_features))
        
        for i in range(1, n_actual_segments + 1):
            mask = (segments == i)
            if np.any(mask):
                superpixel_features[i] = pixel_features[mask].mean(axis=0)
        
        # Remove index 0 (background/unused)
        train_data = superpixel_features[1:]
        
        # 4. Clustering (K-Means on OBJECTS, not pixels)
        if self.verbose:
            print(f"4. Clustering Objects into {n_clusters} Land Use Classes...")
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        object_labels = kmeans.fit_predict(train_scaled)
        
        # 5. Reconstruct Classification Map
        if self.verbose:
            print("5. Reconstructing Classification Map...")
        
        # Map object labels back to pixels
        label_lookup = np.zeros(n_actual_segments + 1, dtype=np.int32)
        label_lookup[1:] = object_labels
        final_map = label_lookup[segments]
        
        # 6. Generate Visualization Data
        vis_boundary = mark_boundaries(image, segments)
        
        # PCA of object features for visualization
        pca = PCA(n_components=3)
        pca_feats = pca.fit_transform(train_scaled)
        
        # Map PCA colors back to segments
        pca_map = np.zeros((h, w, 3), dtype=np.float32)
        pca_norm = (pca_feats - pca_feats.min()) / (pca_feats.max() - pca_feats.min())
        
        for ch in range(3):
            lookup_pca = np.zeros(n_actual_segments + 1, dtype=np.float32)
            lookup_pca[1:] = pca_norm[:, ch]
            pca_map[:, :, ch] = lookup_pca[segments]
        
        return final_map, vis_boundary, pca_map
