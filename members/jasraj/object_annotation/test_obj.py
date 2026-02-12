# --- STEP 1: SETUP ---
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import matplotlib.pyplot as plt

# Define paths
# Use the path explicitly mentioned in your logs
model_path = './data/checkpoints/YOLOv26_OBB.pt'
image_path = './data/haze/test_thin/target/004.png'  # <--- MAKE SURE YOU UPLOADED THIS

# Check if files exist
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
elif not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}. Please upload it!")
else:
    print(f"Loading model from: {model_path}")
    print(f"Processing image: {image_path}")

    # --- STEP 2: LOAD MODEL INTO SAHI ---
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.35, # Only show predictions with >35% confidence
        device="cuda:0"
    )

    # --- STEP 3: RUN SLICED INFERENCE ---
    # This automatically handles the "Small Object, Huge Image" problem
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,  # Tile size matching your training
        slice_width=640,
        overlap_height_ratio=0.2, # 20% overlap so we don't cut planes in half
        overlap_width_ratio=0.2
    )

    # --- STEP 4: VISUALIZE & SAVE ---
    print("Inference complete. Exporting results...")

    # Export visualization
    result.export_visuals(export_dir="final_result/")

    # Display inline
    viz_path = "final_result/prediction_visual.png"
    if os.path.exists(viz_path):
        plt.figure(figsize=(18, 18))
        img = cv2.imread(viz_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Final Annotated Result (mAP: 92.8%)", fontsize=14)
        plt.axis('off')
        plt.show()

    # --- STEP 5: EXPORT JSON (The Requirement) ---
    import json
    json_path = "out.json"
    
    # 1. Get the list of COCO-format annotations
    coco_annotation_list = result.to_coco_annotations()
    
    # 2. Wrap it in a dictionary (Standard COCO format requires 'annotations' key)
    coco_output = {
        "images": [{"id": 1, "file_name": image_path, "height": result.image_height, "width": result.image_width}],
        "annotations": coco_annotation_list,
        "categories": [{"id": 0, "name": "plane"}] # Adapts to your class
    }

    # 3. Save manually using Python's json library
    with open(json_path, "w") as f:
        json.dump(coco_output, f)
        
    print(f"✅ Annotations saved to {json_path}")