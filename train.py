# =============================================================================
# train/train.py — YOLOv8 pothole detection training script
# =============================================================================

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

# =============================================================================
# Install & Import Roboflow
# =============================================================================
try:
    from roboflow import Roboflow
except ImportError:
    os.system("pip install roboflow")
    from roboflow import Roboflow

# =============================================================================
# Roboflow Dataset Download
# =============================================================================
def download_dataset():
    rf = Roboflow(api_key="jfHsSqvQtVBFQXObxzWR")
    project = rf.workspace("nidhis-workspace-m3isz").project("project_13-qpowz")
    version = project.version(1)
    dataset = version.download("yolov8")
    return os.path.join(dataset.location, "data.yaml")

# =============================================================================
# Configuration — edit these before running
# =============================================================================

DATASET_YAML   = download_dataset()   # updated here
BASE_MODEL     = "yolov8n.pt"
PROJECT_NAME   = "runs/detect"
EXPERIMENT     = "pothole_v1"
EPOCHS         = 50
IMAGE_SIZE     = 640
BATCH_SIZE     = 16
WORKERS        = 4
DEVICE         = "0"

# =============================================================================
# Pre-flight checks
# =============================================================================

def check_dataset(yaml_path: str):
    if not os.path.exists(yaml_path):
        print(f"[ERROR] Dataset YAML not found: {yaml_path}")
        print("  Download from Roboflow → Export → YOLOv8 format")
        sys.exit(1)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    print(f"[dataset] Classes: {data.get('names', [])}")
    print(f"[dataset] Train:   {data.get('train', 'N/A')}")
    print(f"[dataset] Val:     {data.get('val',   'N/A')}")

    names = data.get("names", [])
    if isinstance(names, dict):
        names = list(names.values())
    has_pothole = any("pothole" in n.lower() for n in names)
    if not has_pothole:
        print(f"[WARNING] No 'pothole' class found in {names}. Check dataset.")

    return data

# =============================================================================
# Training
# =============================================================================

def train():
    print("=" * 60)
    print(" YOLOv8 Pothole Detection Training")
    print("=" * 60)

    check_dataset(DATASET_YAML)

    model = YOLO(BASE_MODEL)

    results = model.train(
        data        = DATASET_YAML,
        epochs      = EPOCHS,
        imgsz       = IMAGE_SIZE,
        batch       = BATCH_SIZE,
        workers     = WORKERS,
        device      = DEVICE,
        project     = PROJECT_NAME,
        name        = EXPERIMENT,
        exist_ok    = True,

        augment     = True,
        fliplr      = 0.5,
        flipud      = 0.0,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 5.0,
        translate   = 0.1,
        scale       = 0.5,
        mosaic      = 1.0,
        mixup       = 0.1,

        optimizer   = "AdamW",
        lr0         = 0.001,
        lrf         = 0.01,
        momentum    = 0.937,
        weight_decay= 0.0005,

        patience    = 15,

        save        = True,
        save_period = -1,
        plots       = True,
    )

    best_pt = Path(PROJECT_NAME) / EXPERIMENT / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print(f" Training complete.")
    print(f" Best model: {best_pt}")
    print(f" mAP50:      {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f" mAP50-95:   {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print("=" * 60)
    print()
    print(" NEXT STEP:")
    print(f"   cp {best_pt} ../flask-ml-service/model/best.pt")
    print()

    return best_pt

# =============================================================================
# Severity threshold calibration
# =============================================================================

def calibrate_severity(best_pt_path: str, dataset_yaml: str):
    import numpy as np

    print("\n Calibrating severity thresholds...")

    model  = YOLO(best_pt_path)
    data   = check_dataset(dataset_yaml)

    val_path = data.get("val", "")
    if not os.path.exists(val_path):
        print(f"[SKIP] Val path not found: {val_path}")
        return

    results = model.predict(
        source  = val_path,
        conf    = 0.40,
        imgsz   = IMAGE_SIZE,
        verbose = False,
        stream  = True,
    )

    area_ratios = []
    for r in results:
        img_area = r.orig_shape[0] * r.orig_shape[1]
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox_area = (x2 - x1) * (y2 - y1)
            area_ratios.append(bbox_area / img_area)

    if not area_ratios:
        print("[SKIP] No detections found on val set.")
        return

    ratios = np.array(area_ratios)
    print(f"\n Bbox area / image area — distribution over {len(ratios)} detections:")
    print(f"   min:    {ratios.min():.4f}")
    print(f"   p25:    {np.percentile(ratios, 25):.4f}")
    print(f"   median: {np.percentile(ratios, 50):.4f}")
    print(f"   p75:    {np.percentile(ratios, 75):.4f}")
    print(f"   max:    {ratios.max():.4f}")
    print()
    print(" Suggested severity thresholds:")
    print(f"   SEVERITY_LOW_MAX    = {np.percentile(ratios, 33):.3f}")
    print(f"   SEVERITY_MEDIUM_MAX = {np.percentile(ratios, 66):.3f}")
    print()

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    best_pt = train()

    run_calibration = "--calibrate" in sys.argv
    if run_calibration:
        calibrate_severity(str(best_pt), DATASET_YAML)