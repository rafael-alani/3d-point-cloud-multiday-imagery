"""
Core functionality for image restoration.

This module implements satellite image restoration techniques:
- Non-Local Means (NLM) denoising
- Dark Channel Prior (DCP) dehazing
- Unsharp mask sharpening
"""

import cv2
import numpy as np


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


# --- Restoration Engine ---

class RestorationEngine:
    """
    Image restoration engine for satellite imagery.
    
    Implements a three-stage pipeline:
    1. Non-Local Means denoising
    2. Dark Channel Prior dehazing
    3. Unsharp mask sharpening
    """
    
    def __init__(self):
        """Initialize the restoration engine."""
        pass
    
    def add_degradation(self, image):
        """
        Simulate atmospheric degradation (haze + noise) for testing.
        
        This is useful for creating test datasets from clean images.
        
        Args:
            image: Clean RGB image (uint8)
            
        Returns:
            Degraded image with haze and noise
        """
        img_float = image.astype(np.float32) / 255.0
        h, w, c = img_float.shape
        
        # Simulate atmospheric haze
        A = 0.85  # Atmospheric light intensity
        
        # Random transmission map (varies spatially)
        tx = np.random.uniform(0.5, 0.9, (h // 10, w // 10))
        tx = cv2.resize(tx, (w, h), interpolation=cv2.INTER_CUBIC)
        tx = np.clip(tx, 0.1, 1.0)
        tx = np.dstack([tx] * 3)
        
        # Apply haze model: I(x) = J(x)t(x) + A(1-t(x))
        hazy = img_float * tx + A * (1 - tx)
        
        # Add Gaussian noise
        noise = np.random.normal(0, 0.03, hazy.shape)
        
        return np.clip((hazy + noise) * 255, 0, 255).astype(np.uint8)
    
    def get_dark_channel(self, image, patch_size=25):
        """
        Compute the dark channel of an image.
        
        The dark channel prior observes that in most outdoor images,
        at least one color channel has very low intensity in most patches.
        Haze-free regions have low dark channel values.
        
        Args:
            image: RGB image (float, 0-1 range)
            patch_size: Size of local patch (default: 25)
            
        Returns:
            Dark channel (grayscale)
        """
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        return cv2.erode(min_channel, kernel)
    
    def run_denoising_nlm(self, image):
        """
        Apply Non-Local Means (NLM) denoising.
        
        NLM is an advanced denoising algorithm that compares patches
        across the image to preserve edges while removing noise.
        
        Args:
            image: RGB image (uint8)
            
        Returns:
            Denoised image
        """
        return cv2.fastNlMeansDenoisingColored(
            image, None,
            h=6,                    # Filter strength for luminance
            hColor=6,               # Filter strength for color
            templateWindowSize=7,   # Patch size
            searchWindowSize=21     # Search window size
        )
    
    def run_dehazing_dcp(self, image, omega=0.95):
        """
        Apply Dark Channel Prior (DCP) dehazing.
        
        This implements the He et al. (2011) dark channel prior algorithm
        for single image haze removal. It estimates atmospheric light and
        transmission map to recover the haze-free scene.
        
        Args:
            image: Hazy RGB image (uint8)
            omega: Haze retention parameter (default: 0.95)
                   Lower values retain more haze
            
        Returns:
            Tuple of (dehazed_image, transmission_map)
        """
        img_float = image.astype(np.float32) / 255.0
        
        # 1. Estimate atmospheric light
        dark = self.get_dark_channel(img_float)
        
        # Pick top 0.1% brightest pixels in dark channel
        num_pixels = dark.size
        num_top = int(max(num_pixels * 0.001, 1))
        indices = np.argpartition(dark.ravel(), -num_top)[-num_top:]
        
        # Use median of corresponding pixels as atmospheric light
        flat_img = img_float.reshape(-1, 3)
        A_est = np.median(flat_img[indices], axis=0)
        
        # 2. Estimate transmission map
        norm_img = img_float / (A_est + 1e-6)
        dark_norm = self.get_dark_channel(norm_img)
        transmission = 1 - omega * dark_norm
        transmission = np.clip(transmission, 0.1, 1.0)
        
        # Refine transmission with guided filter (approximated by Gaussian blur)
        transmission = cv2.GaussianBlur(transmission, (21, 21), 0)
        t_stack = np.dstack([transmission] * 3)
        
        # 3. Recover scene radiance
        # J(x) = (I(x) - A) / t(x) + A
        J = (img_float - A_est) / t_stack + A_est
        
        return np.clip(J * 255, 0, 255).astype(np.uint8), transmission
    
    def run_sharpening(self, image):
        """
        Apply unsharp mask sharpening.
        
        This enhances edges and fine details by subtracting a blurred
        version of the image from itself.
        
        Args:
            image: RGB image (uint8)
            
        Returns:
            Sharpened image
        """
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharp
    
    def restore(self, image, denoise=True, dehaze=True, sharpen=True):
        """
        Run the complete restoration pipeline.
        
        Args:
            image: Degraded RGB image (uint8)
            denoise: Apply denoising (default: True)
            dehaze: Apply dehazing (default: True)
            sharpen: Apply sharpening (default: True)
            
        Returns:
            Dictionary with keys:
            - 'final': Final restored image
            - 'denoised': After denoising stage
            - 'dehazed': After dehazing stage
            - 'transmission': Transmission map from dehazing
        """
        result = {'original': image.copy()}
        current = image.copy()
        
        # Stage 1: Denoising
        if denoise:
            current = self.run_denoising_nlm(current)
            result['denoised'] = current.copy()
        
        # Stage 2: Dehazing
        transmission = None
        if dehaze:
            current, transmission = self.run_dehazing_dcp(current)
            result['dehazed'] = current.copy()
            result['transmission'] = transmission
        
        # Stage 3: Sharpening
        if sharpen:
            current = self.run_sharpening(current)
        
        result['final'] = current
        return result
