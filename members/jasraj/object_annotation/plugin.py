from interface import SatellitePlugin
import numpy as np
import warnings
from .core import ObjectDetector

# Suppress numpy 'where' warning from napari's internal shape rendering
# This warning comes from napari's triangulation code, not our code
warnings.filterwarnings('ignore', message="'where' used without 'out'", category=UserWarning)

class ObjectAnnotationPlugin(SatellitePlugin):
    """
    Object detection using YOLOv26-OBB with SAHI sliced inference.
    Designed for detecting small objects (e.g., planes) in large satellite images.
    """
    
    def __init__(self, model_path='./data/checkpoints/YOLOv26_OBB.pt', confidence_threshold=0.35):
        """
        Args:
            model_path: Path to YOLO model weights (default: ./YOLOv26_OBB.pt)
            confidence_threshold: Minimum confidence for detections (default: 0.35)
        """
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device='cpu'  # Change to 'cuda:0' if GPU available
        )
    
    @property
    def name(self):
        return "Object Detection (YOLO + SAHI)"
    
    def run(self, image: np.ndarray):
        """
        Run object detection on input image using sliced inference.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            List of napari layers with detected objects as shapes
        """
        # Check if SAHI is available
        if not self.detector.sahi_available:
            print("Error: SAHI library not installed. Install with: pip install sahi")
            return [(image, {"name": "[Object Detection] Error: SAHI not installed", "rgb": True}, "image")]
        
        # Ensure image is uint8 RGB
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        
        print(f"Running object detection with {self.detector.model_path}...")
        print("Using sliced inference (640x640 tiles, 20% overlap)...")
        
        # Run detection
        result = self.detector.detect(
            image,
            slice_height=640,
            slice_width=640,
            overlap_ratio=0.2
        )
        
        if result is None:
            return [(image, {"name": "[Object Detection] Failed - check console", "rgb": True}, "image")]
        
        print(f"Detection complete! Found {len(result.object_prediction_list)} objects")
        
        # Convert to napari shapes
        shapes_data, properties = self.detector.result_to_shapes(result)
        
        # Export to JSON
        self.detector.export_to_coco_json(result, image.shape[:2], 'out.json')
        
        if len(shapes_data) == 0:
            print("No objects detected.")
            return [(image, {"name": "[Object Detection] No detections found", "rgb": True}, "image")]
        
        # Create shape types list (all polygons)
        shape_types = ['polygon'] * len(shapes_data)
        
        return [
            (image, {"name": "[Object Detection] Input", "rgb": True}, "image"),
            (shapes_data, {
                "name": f"[Object Detection] Output (n={len(shapes_data)})",
                "shape_type": shape_types,
                "edge_color": 'lime',
                "edge_width": 2,
                "face_color": 'transparent',
                "properties": properties,
                "text": {
                    'string': '{class}: {confidence:.2f}',
                    'size': 10,
                    'color': 'lime'
                }
            }, "shapes")
        ]
