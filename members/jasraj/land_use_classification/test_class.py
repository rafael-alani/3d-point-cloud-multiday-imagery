"""
Test script for Land Use Classification using OBIA.

This script demonstrates the OBIA classifier on satellite imagery.
"""

import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from .core import OBIAClassifier, normalize_band, smart_crop

def test_land_use(image_path, n_clusters=4):
    print(f"Loading: {image_path}")
    
    # Load Image
    if image_path.lower().endswith(('.tif', '.tiff')):
        data = tifffile.imread(image_path)
        if data.ndim == 3 and data.shape[0] < data.shape[2]: data = np.moveaxis(data, 0, -1)
        if data.ndim == 3 and data.shape[2] >= 3:
            img_disp = np.dstack([normalize_band(data[:,:,i]) for i in [3,2,1]]) if data.shape[2]>=4 else np.dstack([normalize_band(data[:,:,i]) for i in range(3)])
        else:
            img_disp = normalize_band(data)
    else:
        data = cv2.imread(image_path)
        img_disp = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    img_disp = smart_crop(img_disp)
    
    # Run OBIA
    engine = OBIAClassifier()
    # n_segments=1500 gives fine-grained objects. Decrease to 500 for coarser blocks.
    class_map, boundaries, pca_vis = engine.run_obia_pipeline(img_disp, n_segments=1500, n_clusters=n_clusters)
    
    # --- Visualization ---
    plt.figure(figsize=(18, 10))
    
    # 1. Input with Boundaries
    plt.subplot(2, 3, 1)
    plt.imshow(boundaries)
    plt.title("1. Superpixel Segmentation (SLIC)\n(Pre-processing Step)")
    plt.axis('off')
    
    # 2. PCA Features
    plt.subplot(2, 3, 2)
    plt.imshow(pca_vis)
    plt.title("2. Object Features (PCA)\n(Aggregated Color+Texture+Entropy)")
    plt.axis('off')
    
    # 3. Final Map
    plt.subplot(2, 3, 3)
    plt.imshow(class_map, cmap='tab10', interpolation='nearest')
    plt.title(f"3. Final OBIA Classification\n(K={n_clusters})")
    plt.axis('off')
    
    # 4. Zoomed Comparison
    h, w, _ = img_disp.shape
    cy, cx = h//2, w//2
    sl = (slice(cy-100, cy+100), slice(cx-100, cx+100))
    
    plt.subplot(2, 3, 4)
    plt.imshow(img_disp[sl])
    plt.title("Zoom: Input")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(boundaries[sl])
    plt.title("Zoom: Segments")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(class_map[sl], cmap='tab10', interpolation='nearest')
    plt.title("Zoom: Classification")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = "/Users/jasraj/Desktop/finalproject_16/data/RGB-PanSharpen/RGB-PanSharpen_AOI_5_Khartoum_img31.tif"
    # Try increasing K to 5 or 6 to separate more subtle classes
    test_land_use(path, n_clusters=5)