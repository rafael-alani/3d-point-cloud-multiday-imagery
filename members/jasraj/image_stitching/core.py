"""
Core functionality for advanced image stitching.

This module implements satellite image stitching using:
- SIFT feature detection
- Spatial consistency matching (GMS-like filtering)
- Homography estimation with RANSAC
- Optimal seam finding via dynamic programming
"""

import cv2
import numpy as np
from scipy.spatial import cKDTree


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


# --- Feature Matching with Spatial Consistency ---

def match_features_spatial(kps1, des1, kps2, des2, radius=50.0, min_neighbors=3):
    """
    Advanced feature matching with spatial consistency filtering (GMS-like).
    
    A match is only valid if its spatial neighbors also match geometrically,
    ensuring global geometric consistency.
    
    Args:
        kps1: Keypoints from image 1
        des1: Descriptors from image 1
        kps2: Keypoints from image 2
        des2: Descriptors from image 2
        radius: Pixel radius for neighborhood search (default: 50.0)
        min_neighbors: Minimum number of consistent neighbors (default: 3)
        
    Returns:
        List of spatially consistent matches
    """
    # 1. Initial KNN matching with ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    initial_good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            initial_good.append(m)
    
    if len(initial_good) < 10:
        return []
    
    # 2. Extract match coordinates
    pts1 = np.float32([kps1[m.queryIdx].pt for m in initial_good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in initial_good])
    
    # 3. Spatial consistency verification
    # build KD-trees for fast spatial queries
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    
    spatial_good = []
    
    for i, m in enumerate(initial_good):
        # Find spatial neighbors in both images
        neighbors1 = tree1.query_ball_point(pts1[i], radius)
        neighbors2 = tree2.query_ball_point(pts2[i], radius)
        
        # Count how many neighbors are consistent
        # (i.e., appear in both neighborhood sets)
        common = len(set(neighbors1) & set(neighbors2))
        
        # Keep match if enough neighbors agree
        if common >= min_neighbors:
            spatial_good.append(m)
    
    print(f"   - Raw Matches: {len(initial_good)} -> Spatially Consistent: {len(spatial_good)}")
    return spatial_good


# --- Optimal Seam Finding ---

def find_optimal_seam(img_ref, img_warped, mask_overlap):
    """
    Find optimal seam between two overlapping images using dynamic programming.
    
    This implements a seam-carving-like algorithm to find the minimum-energy
    path through the overlap region, minimizing visual artifacts.
    
    Args:
        img_ref: Reference image (warped to common canvas)
        img_warped: Image to blend (warped to common canvas)
        mask_overlap: Binary mask of overlap region
        
    Returns:
        Seam mask (255 = use img_ref, 0 = use img_warped)
    """
    # 1. Compute pixel-wise difference energy
    diff = cv2.absdiff(img_ref, img_warped)
    energy = np.sum(diff, axis=2)  # Sum across color channels
    
    # Force seam to stay within overlap region
    energy[mask_overlap == 0] = 10000
    
    h, w = energy.shape
    
    # 2. Dynamic programming: accumulate minimum energy paths
    dp = np.zeros_like(energy, dtype=np.float32)
    dp[0, :] = energy[0, :]
    
    # Track parent pointers for path reconstruction
    parent = np.zeros((h, w), dtype=np.int32)
    
    for y in range(1, h):
        for x in range(w):
            # Check three possible parents: above-left, above, above-right
            prev_x_start = max(0, x - 1)
            prev_x_end = min(w, x + 2)
            
            neighbors = dp[y - 1, prev_x_start:prev_x_end]
            min_idx = np.argmin(neighbors)
            parent_x = prev_x_start + min_idx
            
            # Accumulate energy
            dp[y, x] = energy[y, x] + dp[y - 1, parent_x]
            parent[y, x] = parent_x
    
    # 3. Backtrack to find optimal seam
    path_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Start from bottom row, minimum energy position
    curr_x = np.argmin(dp[-1, :])
    
    # Trace path upward, marking left side as 255
    for y in range(h - 1, -1, -1):
        path_mask[y, :curr_x] = 255
        curr_x = parent[y, curr_x]
    
    return path_mask


# --- Image Stitching Pipeline ---

class ImageStitcher:
    """
    Advanced image stitcher for satellite imagery.
    
    Features:
    - SIFT feature detection
    - Spatial consistency filtering
    - RANSAC homography estimation
    - Optimal seam blending
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the image stitcher.
        
        Args:
            verbose: Print progress messages (default: True)
        """
        self.verbose = verbose
        self.sift = cv2.SIFT_create()
    
    def stitch(self, img1, img2):
        """
        Stitch two overlapping images together.
        
        Args:
            img1: First image (RGB, uint8)
            img2: Second image (RGB, uint8)
            
        Returns:
            Stitched result as numpy array, or None if stitching fails
        """
        if self.verbose:
            print("Detecting SIFT features...")
        
        # 1. Feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        if self.verbose:
            print(f"   Found {len(kp1)} and {len(kp2)} keypoints")
        
        # 2. Feature matching with spatial consistency
        if self.verbose:
            print("Matching with spatial consistency filter...")
        
        matches = match_features_spatial(kp1, des1, kp2, des2)
        
        if len(matches) < 4:
            print("Error: Not enough matches found (need at least 4)")
            return None
        
        # 3. Compute homography
        if self.verbose:
            print(f"Computing homography from {len(matches)} matches...")
        
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print("Error: Homography estimation failed")
            return None
        
        # 4. Calculate output canvas size
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        corners_2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners_2 = cv2.perspectiveTransform(corners_2, H)
        corners_1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corners_1, warped_corners_2), axis=0)
        
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation_dist = [-xmin, -ymin]
        
        # 5. Warp images to common canvas
        if self.verbose:
            print("Warping images to common canvas...")
        
        H_translation = np.array([
            [1, 0, translation_dist[0]],
            [0, 1, translation_dist[1]],
            [0, 0, 1]
        ])
        H_final = H_translation.dot(H)
        output_shape = (xmax - xmin, ymax - ymin)
        
        warped_img2 = cv2.warpPerspective(img2, H_final, output_shape)
        M_trans = np.float32([
            [1, 0, translation_dist[0]],
            [0, 1, translation_dist[1]]
        ])
        warped_img1 = cv2.warpAffine(img1, M_trans, output_shape)
        
        # 6. Create masks for overlap detection
        mask1 = cv2.warpAffine(
            np.ones((h1, w1), dtype=np.uint8) * 255,
            M_trans,
            output_shape
        )
        mask2 = cv2.warpPerspective(
            np.ones((h2, w2), dtype=np.uint8) * 255,
            H_final,
            output_shape
        )
        overlap = cv2.bitwise_and(mask1, mask2)
        
        # 7. Optimal seam blending
        if cv2.countNonZero(overlap) > 0:
            if self.verbose:
                print("Computing optimal seam...")
            
            seam_mask = find_optimal_seam(warped_img1, warped_img2, overlap)
            
            # Compose final result using seam mask
            result = warped_img2.copy()
            final_mask1 = cv2.bitwise_and(mask1, seam_mask)
            result[final_mask1 > 0] = warped_img1[final_mask1 > 0]
        else:
            # No overlap, simple addition
            result = cv2.addWeighted(warped_img1, 1, warped_img2, 1, 0)
        
        # 8. Crop to valid region
        if self.verbose:
            print("Cropping result...")
        
        coords = np.argwhere(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            result = result[y_min:y_max, x_min:x_max]
        
        if self.verbose:
            print("Stitching complete!")
        
        return result
