"""
Core functionality for object detection using YOLO with SAHI.

This module implements oriented bounding box (OBB) object detection
optimized for small objects in large satellite images using:
- YOLOv26-OBB model
- SAHI (Slicing Aided Hyper Inference) for handling large images
- JSON export in COCO format
"""

import os
import json
import tempfile
import cv2
import numpy as np


class ObjectDetector:
    """
    Object detector using YOLOv26-OBB with SAHI sliced inference.
    
    Optimized for detecting small objects (e.g., planes, vehicles, buildings)
    in large satellite images by processing the image in overlapping tiles.
    """
    
    def __init__(self, model_path='./data/checkpoints/YOLOv26_OBB.pt', confidence_threshold=0.35, device='cpu'):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to YOLO model weights (default: ./models/YOLOv26_OBB.pt)
            confidence_threshold: Minimum confidence for detections (default: 0.35)
            device: Device to run inference on ('cpu' or 'cuda:0')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        
        # Check dependencies
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            self.AutoDetectionModel = AutoDetectionModel
            self.get_sliced_prediction = get_sliced_prediction
            self.sahi_available = True
        except ImportError:
            print("Warning: SAHI library not installed. Install with: pip install sahi")
            self.sahi_available = False
        
        # Check model file
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            self._list_available_models()
    
    def _list_available_models(self):
        """List available .pt model files in current directory."""
        print("Available model files:")
        for f in os.listdir('.'):
            if f.endswith('.pt'):
                print(f"  - {f}")
    
    def load_model(self):
        """
        Load the YOLO model into memory.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.sahi_available:
            return False
        
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found: {self.model_path}")
            return False
        
        try:
            self.model = self.AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold,
                device=self.device
            )
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect(self, image, slice_height=640, slice_width=640, overlap_ratio=0.2):
        """
        Run object detection on an image using sliced inference.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            slice_height: Height of each slice (default: 640)
            slice_width: Width of each slice (default: 640)
            overlap_ratio: Overlap between slices (default: 0.2 = 20%)
            
        Returns:
            SAHI prediction result object, or None if detection fails
        """
        if not self.sahi_available:
            print("Error: SAHI not available")
            return None
        
        # Load model if not already loaded
        if self.model is None:
            if not self.load_model():
                return None
        
        # SAHI expects a file path, so save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            # Convert RGB to BGR for cv2
            cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            # Run sliced prediction
            result = self.get_sliced_prediction(
                tmp_path,
                self.model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio
            )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return result
            
        except Exception as e:
            print(f"Error during detection: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None
    
    def result_to_shapes(self, result):
        """
        Convert SAHI prediction result to napari shapes format.
        
        Args:
            result: SAHI prediction result object
            
        Returns:
            Tuple of (shapes_data, properties) where:
            - shapes_data: List of polygons (rectangles)
            - properties: Dict with confidence, class, class_id
        """
        shapes_data = []
        properties = {
            'confidence': [],
            'class': [],
            'class_id': []
        }
        
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            xmin, ymin, xmax, ymax = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            
            # Create rectangle as polygon (napari format: [y, x])
            rect = np.array([
                [ymin, xmin],
                [ymin, xmax],
                [ymax, xmax],
                [ymax, xmin]
            ])
            
            shapes_data.append(rect)
            properties['confidence'].append(pred.score.value)
            properties['class'].append(pred.category.name)
            properties['class_id'].append(pred.category.id)
        
        return shapes_data, properties
    
    def export_to_coco_json(self, result, image_shape, output_path='out.json'):
        """
        Export detection results to COCO JSON format.
        
        Args:
            result: SAHI prediction result object
            image_shape: Tuple of (height, width)
            output_path: Path to save JSON file (default: out.json)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            coco_annotation_list = result.to_coco_annotations()
            
            # Get unique categories
            categories = {}
            for pred in result.object_prediction_list:
                cat_id = pred.category.id
                cat_name = pred.category.name
                if cat_id not in categories:
                    categories[cat_id] = cat_name
            
            coco_output = {
                "images": [{
                    "id": 1,
                    "file_name": "detection_input",
                    "height": image_shape[0],
                    "width": image_shape[1]
                }],
                "annotations": coco_annotation_list,
                "categories": [
                    {"id": cat_id, "name": cat_name}
                    for cat_id, cat_name in categories.items()
                ]
            }
            
            # with open(output_path, "w") as f:
            #     json.dump(coco_output, f, indent=2)
            
            # print(f"✅ Annotations exported to {output_path}")
            # return True
            
        except Exception as e:
            print(f"Warning: Could not export JSON: {e}")
            return False
    
    def detect_and_export(self, image, output_json='out.json', **kwargs):
        """
        Run detection and export results in one call.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            output_json: Path to save JSON file (default: out.json)
            **kwargs: Additional arguments for detect() method
            
        Returns:
            SAHI prediction result object, or None if detection fails
        """
        result = self.detect(image, **kwargs)
        
        if result is not None:
            self.export_to_coco_json(result, image.shape[:2], output_json)
        
        return result
