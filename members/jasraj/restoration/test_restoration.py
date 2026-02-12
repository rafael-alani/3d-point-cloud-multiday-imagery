"""
Test script for Image Restoration using denoise/dehaze/sharpen pipeline.

This script demonstrates the restoration engine on degraded satellite imagery.
"""

import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
from .core import RestorationEngine, normalize_band, smart_crop

def test_restoration(image_path, ground_truth_path=None):
    engine = RestorationEngine()
    
    # Helper function to load and process image
    def load_and_process_image(path, description):
        print(f"Loading {description}: {path}")
        
        # 1. Load Data
        if path.lower().endswith(('.tif', '.tiff')):
            data = tifffile.imread(path)
            is_satellite_data = True
        else:
            data = cv2.imread(path)
            if data is None:
                raise FileNotFoundError(f"Could not load {description} from: {path}")
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            is_satellite_data = False
        
        # 2. Handle Band Selection & Normalization
        # CASE A: Satellite Data (Needs Normalization)
        if is_satellite_data:
            if data.ndim == 3 and data.shape[0] < data.shape[2]: 
                data = np.moveaxis(data, 0, -1)

            if data.ndim == 3 and data.shape[2] >= 4:
                print(f"Using Bands [3, 2, 1] (RGB) for {description}")
                red   = normalize_band(data[:, :, 3])
                green = normalize_band(data[:, :, 2])
                blue  = normalize_band(data[:, :, 1])
                img_8bit = np.dstack([red, green, blue])
            else:
                print(f"Using Standard RGB for {description}")
                img = data[:, :, :3]
                img_8bit = np.dstack([normalize_band(img[:,:,i]) for i in range(3)])
                
            return smart_crop(img_8bit)

        # CASE B: Standard Image (PNG/JPG) - DO NOT NORMALIZE
        else:
            print(f"Standard Image Detected: Skipping Normalization for {description}")
            return data.astype(np.uint8)
    
    # --- LOGIC BRANCHING ---
    
    if ground_truth_path is not None:
        # SCENARIO B: REAL BENCHMARK
        # User provided two files. 
        # File 1 = The Degraded Input (Hazy)
        # File 2 = The Ground Truth (Clean)
        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
            
        degraded = load_and_process_image(image_path, "Input (Degraded)")
        ground_truth = load_and_process_image(ground_truth_path, "Ground Truth")
        
    else:
        # SCENARIO A: SIMULATION
        # User provided only one file.
        # File 1 = The Clean Ground Truth
        # Script must SIMULATE the degradation.
        ground_truth = load_and_process_image(image_path, "Input (Ground Truth)")
        print("Simulating Degradation (Haze + Noise)...")
        degraded = engine.add_degradation(ground_truth)
    
    # --- PIPELINE ---
    print("Running Restoration Pipeline...")
    
    # 1. Denoising
    # denoised = degraded
    denoised = engine.run_denoising_nlm(degraded)
    
    # 2. Dehazing
    dehazed, _ = engine.run_dehazing_dcp(degraded)
    
    # 3. Sharpening
    final_result = engine.run_sharpening(dehazed)
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(16, 10))
    
    # Adjust Zoom Crop
    h, w, _ = ground_truth.shape
    cy, cx = h//2, w//2
    # Ensure slice is within bounds
    start_y, end_y = max(0, cy-60), min(h, cy+60)
    start_x, end_x = max(0, cx-60), min(w, cx+60)
    sl = (slice(start_y, end_y), slice(start_x, end_x))
    
    titles = ["1. Input (Degraded)", "2. Denoised", "3. Dehazed", "4. Final (Sharpened)", "Ground Truth"]
    images = [degraded, denoised, dehazed, final_result, ground_truth]
    
    for i in range(5):
        # ROW 1: Full Image
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i])
        plt.title(f"{titles[i]}\n(Full)", fontsize=9)
        plt.axis('off')

        # ROW 2: Zoomed Crop
        plt.subplot(2, 5, i+6)
        plt.imshow(images[i][sl])
        plt.title(f"Zoom", fontsize=9)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # CASE 1: REAL DATA (Two Paths)
    # Set ground_truth_path to the clean file.
    
    # CASE 2: SIMULATION (One Path)
    # Set ground_truth_path = None. The script will degrade the input file itself.

    input_path = "/Users/jasraj/Desktop/finalproject_16/data/haze/test_thick/input/002.png"
    target_path = "/Users/jasraj/Desktop/finalproject_16/data/haze/test_thick/target/002.png"
    
    # Switch logic based on your needs:
    # Set target_path to None if you want to simulate haze on a clean image.
    test_restoration(
        image_path=input_path,
        ground_truth_path=target_path 
    )