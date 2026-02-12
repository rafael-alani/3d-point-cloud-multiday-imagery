from ultralytics import YOLO
from roboflow import Roboflow
import yaml

# --- 1. DATASET ACQUISITION (The "Big" Dataset) ---
# We use Roboflow to download a specific "Planes Only" subset of DOTA.
# This gives you ~500-1000 real satellite chips, not just 8.
# NOTE: You need a free Roboflow account to get an API KEY.
# If you don't have one, replace this block with "data='DOTAv1.yaml'" (but that is 40GB!)

try:
    print("Downloading DOTA-Planes (OBB) from Roboflow...")
    rf = Roboflow(api_key="Lwr0tV1OaAOYWSEtVnuY") # <--- PASTE KEY HERE
    # This is a public DOTA-plane subset hosted on Roboflow Universe
    project = rf.workspace("class-dvpyb").project("dota-airplane") 
    dataset = project.version(1).download("yolov8-obb")
    dataset_yaml = f"{dataset.location}/data.yaml"
    print(f"Dataset ready at: {dataset_yaml}")
except Exception as e:
    print(f"Roboflow download failed: {e}")
    print("Falling back to DOTAv1.yaml (Warning: Large Download!)")
    dataset_yaml = "DOTAv1.yaml" # Fallback to standard config

# --- 2. HYPERPARAMETER ENGINEERING (The "Scientific" Part) ---
# We don't just use default settings. We tune the "Physics" of the training.
# Satellite objects can appear at ANY angle. Defaults assume "up is up".
aerial_hyp = {
    'degrees': 180.0,      # Full 360-degree rotation (+/- 180)
    'fliplr': 0.5,         # Left-Right Flip
    'flipud': 0.5,         # Up-Down Flip (Crucial for satellite!)
    'shear': 2.5,          # Geometric shear to simulate camera angle distortion
    'mosaic': 1.0,         # Mosaic Augmentation (Stitches 4 images to handle scale)
    'copy_paste': 0.3,     # Copy-Paste (Helps detecting small, crowded planes)
    'optimizer': 'auto',   # YOLO26 auto-selects (often AdamW for transformers)
}

# --- 3. MODEL ARCHITECTURE (The "Research" Part) ---
# YOLO26n-obb is NMS-Free (End-to-End).
model = YOLO("yolo26n-obb.pt") 

# --- 4. LONG TRAINING RUN ---
# We train for 100 epochs. On a T4 GPU, this might take 1-2 hours.
# This satisfies your "takes long to train" requirement.
results = model.train(
    data=dataset_yaml,     
    epochs=100,            # Sufficient convergence for 1000 images
    imgsz=1024,            # High-Res training (Crucial for small satellite objects)
    batch=4,               # Small batch to fit 1024px images in Colab RAM
    patience=20,           # Stop if no improvement for 20 epochs
    name='yolo26_dota_planes',
    **aerial_hyp           # Inject our aerial physics
)

# --- 5. VALIDATION & REPORTING ---
# Validate on the test split to get scientific metrics (mAP50-95)
metrics = model.val()
print(f"Final OBB mAP50: {metrics.box.map50}")