"""
Test script for Image Stitching using SIFT and optimal seam finding.

This script demonstrates the image stitcher on satellite imagery.
"""

import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from .core import ImageStitcher, normalize_band, smart_crop

# --- HELPER: GENERATE SYNTHETIC OVERLAP (DATASET CREATOR) ---
def generate_test_pair(image_path, overlap_ratio=0.4):
    """
    Takes a single satellite image and splits it into two overlapping, 
    misaligned strips to simulate a real stitching scenario.
    """
    print(f"Loading Ground Truth: {image_path}")
    try:
        if image_path.lower().endswith(('.tif', '.tiff')):
            data = tifffile.imread(image_path)
            if data.ndim == 3 and data.shape[0] < data.shape[2]: data = np.moveaxis(data, 0, -1)
            img = data[:,:,:3] if data.ndim==3 else data
            # Normalize
            img = np.dstack([normalize_band(img[:,:,i]) for i in range(3)])
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = smart_crop(img)
    except Exception as e:
        print(f"Error: {e}")
        return None, None

    h, w = img.shape[:2]
    split_point = int(w * (0.5 + overlap_ratio/2))
    start_point = int(w * (0.5 - overlap_ratio/2))
    
    # Image 1 (Left side)
    img1 = img[:, :split_point].copy()
    
    # Image 2 (Right side)
    img2_orig = img[:, start_point:].copy()
    
    # DISTORTION: Rotate and shift Img2 to make it hard
    h2, w2 = img2_orig.shape[:2]
    # Rotation of 3 degrees, slight scale change
    M = cv2.getRotationMatrix2D((w2//2, h2//2), 3.0, 1.02) 
    M[0, 2] += 10 # Shift X
    M[1, 2] -= 5  # Shift Y
    
    img2 = cv2.warpAffine(img2_orig, M, (w2, h2), borderMode=cv2.BORDER_REFLECT)
    
    return img1, img2

# --- MAIN STITCHING PIPELINE ---
def stitch_satellite_images(path):
    """Main test function using the ImageStitcher class."""
    # 1. Generate test dataset
    img1, img2 = generate_test_pair(path, overlap_ratio=0.4)
    if img1 is None:
        return
    
    # 2. Stitch using the core stitcher
    stitcher = ImageStitcher(verbose=True)
    result = stitcher.stitch(img1, img2)
    
    if result is None:
        print("Stitching failed!")
        return

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title("Strip 1 (Reference)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title("Strip 2 (Distorted)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title("Optimal Seam Stitching")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Use one of your large Khartoum images
    path = "data/RGB-PanSharpen/RGB-PanSharpen_AOI_5_Khartoum_img7.tif"
    stitch_satellite_images(path)