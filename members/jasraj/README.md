# Satellite Image Processing Components

This directory contains four advanced satellite image processing components developed by Jasraj, implementing state-of-the-art computer vision and machine learning techniques for remote sensing applications.

## Overview

The implementation consists of four main components, each addressing a critical aspect of satellite image analysis:

1. **Image Restoration** - Dehazing and denoising pipeline using Color Attenuation Prior (CAP) and Non-Local Means
2. **Image Stitching** - Mosaicking with spatial consistency matching and optimal seam cutting
3. **Land Use Classification** - Unsupervised OBIA pipeline with Gabor filter banks and SLIC superpixels
4. **Object Annotation** - Oriented bounding box detection using YOLO-OBB with SAHI sliced inference

All components are integrated as plugins compatible with the main satellite image processing interface.

## Component Details

### 1. Image Restoration (`restoration/`)

**Purpose**: Remove atmospheric degradation (haze, noise) from satellite imagery to improve visual quality and downstream processing accuracy.

**Implementation**:
- **Dehazing**: Color Attenuation Prior (CAP) method with atmospheric scattering model inversion
- **Denoising**: Non-Local Means (NLM) algorithm for edge-preserving noise reduction
- **Sharpening**: Unsharp mask technique for detail enhancement

**Key Features**:
- Three-stage pipeline: denoising → dehazing → sharpening
- Dark channel prior for atmospheric light estimation
- Transmission map refinement using guided filter approximation
- Handles both simulated and real-world degraded imagery

**Files**:
- `core.py`: `RestorationEngine` class with full pipeline implementation
- `plugin.py`: `RestorationPlugin` for interface integration
- `test_restoration.py`: Testing and visualization script

### 2. Image Stitching (`image_stitching/`)

**Purpose**: Automatically align and blend multiple overlapping satellite images into a seamless mosaic.

**Implementation**:
- **Feature Detection**: SIFT (Scale-Invariant Feature Transform) keypoints
- **Spatial Consistency Matching**: KD-Tree based neighbor verification (GMS-style filtering)
- **Homography Estimation**: RANSAC-based robust transformation computation
- **Optimal Seam Finding**: Dynamic programming for minimum-energy path through overlap region

**Key Features**:
- Spatial consistency filtering reduces false matches by verifying geometric agreement
- Energy minimization matrix computation for seam optimization
- Backtracking algorithm reconstructs optimal seam path
- Automatic canvas size calculation and image warping
- Handles perspective transformations and large image pairs

**Files**:
- `core.py`: `ImageStitcher` class with matching and blending logic
- `plugin.py`: `ImageStitchingPlugin` for interface integration
- `test_stich.py`: Testing script for stitching pipeline

### 3. Land Use Classification (`land_use_classification/`)

**Purpose**: Unsupervised classification of satellite imagery into land use categories using Object-Based Image Analysis (OBIA).

**Implementation**:
- **Superpixel Segmentation**: SLIC (Simple Linear Iterative Clustering) for object generation
- **Feature Extraction**: 
  - Gabor filter bank (16 filters: 4 orientations × 2 scales × 2 wavelengths)
  - Local entropy computation for texture randomness
  - LAB color space transformation for shadow-invariant analysis
- **Feature Aggregation**: Mean feature vector per superpixel (OBIA principle)
- **Clustering**: K-Means on object-level features (not pixel-level)

**Key Features**:
- Gabor bank construction with configurable orientations, scales, and wavelengths
- Multi-modal feature extraction: color (LAB), texture (Gabor), and entropy
- Feature aggregation logic computing statistics per superpixel
- PCA visualization of object feature space
- Configurable number of segments and clusters

**Files**:
- `core.py`: `OBIAClassifier` class with full OBIA pipeline
- `plugin.py`: `LandUseClassificationPlugin` for interface integration
- `test_class.py`: Testing script for classification

### 4. Object Annotation (`object_annotation/`)

**Purpose**: Detect and annotate objects (e.g., aircraft, vehicles, buildings) in large satellite images using deep learning.

**Implementation**:
- **Model**: YOLOv26-OBB (Oriented Bounding Box) architecture
- **Inference Strategy**: SAHI (Slicing Aided Hyper Inference) for large image handling
- **Export Format**: COCO JSON annotation format

**Key Features**:
- SAHI slicing logic: divides large images into overlapping tiles (640×640)
- Handles small object detection in high-resolution imagery
- Automatic overlap management (20% default) to prevent object splitting
- COCO-format JSON export with category and bounding box information
- Confidence threshold filtering

**Files**:
- `core.py`: `ObjectDetector` class with SAHI integration
- `plugin.py`: `ObjectAnnotationPlugin` for interface integration
- `obj_model.py`: Model loading and configuration utilities
- `test_obj.py`: Testing script for object detection
- `models/YOLOv26_OBB.pt`: Pre-trained model weights

## Architecture

All components follow a consistent architecture:

```
component_name/
├── __init__.py          # Module exports
├── core.py              # Core algorithm implementation
├── plugin.py            # Interface plugin wrapper
└── test_*.py            # Testing/validation script
```

The `core.py` files contain the algorithmic implementations, while `plugin.py` files provide the interface compatibility layer. This separation allows for:
- Independent testing of algorithms
- Reusability across different interfaces
- Clear separation of concerns

## Implementation Highlights

### Restoration Component
- **CAP Depth Equation**: Custom implementation of the Color Attenuation Prior depth estimation model
- **Guided Filter**: Implemented from scratch using box filters for transmission map refinement
- **Atmospheric Scattering Inversion**: Full pipeline for recovering scene radiance from hazy images using the physical model: `J(x) = (I(x) - A) / t(x) + A`

### Stitching Component
- **Spatial Neighbor Verification**: GMS-style logic using KD-Tree spatial queries to ensure geometric consistency
- **Energy Minimization Matrix**: Dynamic programming table construction for optimal seam finding
- **Backtracking Algorithm**: Path reconstruction from accumulated energy matrix to determine final seam

### Classification Component
- **Gabor Bank Construction**: Systematic generation of 16 Gabor filters covering multiple orientations, scales, and wavelengths
- **Feature Extraction Logic**: Multi-modal feature computation combining color, texture, and entropy measures
- **Superpixel Aggregation**: Mean feature vector computation per SLIC segment for object-level analysis
- **K-Means Integration**: Clustering on aggregated object features rather than raw pixels

### Annotation Component
- **Training Pipeline**: Custom YOLO-OBB training setup for DOTA dataset subset
- **SAHI Slicing Logic**: Implementation of tile-based inference with overlap management
- **JSON Export Logic**: COCO-format annotation generation with proper category mapping

## Integration

All components are registered in `__init__.py` and can be imported as:

```python
from members.jasraj import (
    RestorationPlugin,
    ImageStitchingPlugin,
    LandUseClassificationPlugin,
    ObjectAnnotationPlugin
)
```

They integrate seamlessly with the main satellite image processing interface through the `SatellitePlugin` base class.

## Testing

All components can be tested using the viewer interface with sample data from `test_data/`. Each component has corresponding test images:

- **Restoration**: `test_data/restoration/` - Single image input
- **Land Use Classification**: `test_data/land_use_classification/` - Single image input
- **Object Annotation**: `test_data/object_annotation/` - Single image input
- **Image Stitching**: `test_data/image_stitching/` - Contains two folders (`Input1/` and `Input2/`) with images in each. The stitching component requires two images to be loaded in the viewer simultaneously.

Load the appropriate test images in the viewer and run the corresponding plugin to test each technique.

## References

### Papers
- **CAP (Color Attenuation Prior)**: Zhu et al. - "A Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior"
- **Guided Filter**: He et al. - "Guided Image Filtering"
- **GMS (Grid-based Motion Statistics)**: Bian et al. - "GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence"
- **Graph Cuts**: Kwatra et al. - "Graphcut Textures: Image and Video Synthesis Using Graph Cuts"

### Libraries
- **OpenCV**: Basic filtering operations, SIFT, homography estimation
- **SciPy**: cKDTree for spatial queries
- **scikit-image**: SLIC superpixels, entropy filters
- **scikit-learn**: KMeans, PCA, StandardScaler
- **Ultralytics**: YOLO architecture
- **SAHI**: Slicing library for large image inference
- **DOTA-v1.0**: Dataset for object detection training

---

## LLM Declaration: AI-Assisted Development

This section documents the role of AI (Large Language Models) in the development of these components, providing transparency about what was implemented manually versus with AI assistance.

### AI Contribution Summary

The AI assistant (LLM) was primarily used for:

1. **Code Structure and Organization**: 
   - Designing the plugin architecture and interface compatibility
   - Organizing code into logical modules (`core.py`, `plugin.py`)
   - Establishing consistent patterns across components

2. **Algorithm Implementation Guidance**:
   - Providing reference implementations for standard algorithms (SIFT matching, RANSAC)
   - Suggesting parameter values and optimization strategies
   - Helping with edge cases and error handling

3. **Code Documentation**:
   - Writing docstrings and comments
   - Creating readme.md

### Manual Implementation (Student Work)

The following core algorithmic components were implemented manually by the student:

#### Restoration Component
- **CAP Depth Equation**: Mathematical implementation of the Color Attenuation Prior model
- **Guided Filter from Scratch**: Box filter-based implementation for transmission map refinement
- **Atmospheric Scattering Inversion Pipeline**: Complete implementation of the physical model `J(x) = (I(x) - A) / t(x) + A` with atmospheric light estimation

#### Stitching Component
- **Spatial Neighbor Verification Logic**: GMS-style algorithm using KD-Tree queries to verify geometric consistency
- **Energy Minimization Matrix**: Dynamic programming table construction for optimal seam finding
- **Backtracking Algorithm**: Path reconstruction logic to determine the final seam from accumulated energy

### Collaboration Model

The development followed a collaborative model where:
- **Student**: Designed algorithms, implemented core mathematical/logical components, and made architectural decisions
- **AI Assistant**: Provided code scaffolding, researching techniques, documentation, and debugging support
- **Result**: A complete, functional implementation where the student's algorithmic contributions are clearly identifiable and the AI's role was supportive rather than generative of the core research contributions

This declaration ensures transparency about the development process while highlighting the substantial manual implementation work completed by the student.
