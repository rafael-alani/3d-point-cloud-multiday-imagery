"""
Jasraj's Satellite Image Processing Components

This module contains four main components:
1. Land Use Classification - OBIA-based classifier using superpixels and texture features
2. Image Stitching - SIFT-based stitching with spatial consistency and optimal seam finding
3. Image Restoration - Denoising, dehazing, and sharpening pipeline
4. Object Annotation - YOLO-based object detection with SAHI sliced inference
"""

from .land_use_classification import LandUseClassificationPlugin
from .image_stitching import ImageStitchingPlugin
from .restoration import RestorationPlugin
from .object_annotation import ObjectAnnotationPlugin

__all__ = [
    'LandUseClassificationPlugin',
    'ImageStitchingPlugin',
    'RestorationPlugin',
    'ObjectAnnotationPlugin'
]
