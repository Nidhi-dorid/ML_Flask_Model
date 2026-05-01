# =============================================================================
# detector.py — YOLOv8 inference engine + severity scoring
# =============================================================================

import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import config

import torch
from ultralytics.nn.tasks import DetectionModel

_model = None

def load_model():
    global _model

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{config.MODEL_PATH}'."
        )

    torch.set_num_threads(1)   # reduce CPU usage

    _model = YOLO(config.MODEL_PATH)
    _model.to("cpu")  # force CPU

    print(f"[detector] Model loaded from {config.MODEL_PATH}")
    return _model


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


# -----------------------------------------------------------------------------
# Severity scoring
# -----------------------------------------------------------------------------

def compute_severity(total_bbox_area: float,
                     image_width: int, image_height: int) -> tuple[str, float]:
    """
    Compute severity based on the SUMMED area of ALL detected pothole bounding
    boxes divided by the total image area.  Using the single largest box caused
    almost every image to land in "Medium" because one pothole rarely covers
    more than 8 % of the frame.
    """
    image_area = image_width * image_height
    if image_area == 0:
        return "low", 0.0

    raw_score = total_bbox_area / image_area
    severity_score = min(round(float(raw_score), 4), 1.0)

    if severity_score < config.SEVERITY_LOW_MAX:
        severity_label = "low"
    elif severity_score < config.SEVERITY_MEDIUM_MAX:
        severity_label = "medium"
    else:
        severity_label = "high"

    return severity_label, severity_score


# -----------------------------------------------------------------------------
# Main detection function
# -----------------------------------------------------------------------------

def detect_pothole(pil_image: Image.Image) -> dict:

    model = get_model()

    # 🔽 Resize first (important)
    MAX_SIZE = 1024
    if max(pil_image.size) > MAX_SIZE:
        pil_image.thumbnail((MAX_SIZE, MAX_SIZE))

    # 🔽 Get correct size AFTER resize
    image_w, image_h = pil_image.size

    # --- Run inference -------------------------------------------------------
    try:
        results = model.predict(
            source=pil_image,
            conf=config.MODEL_CONFIDENCE_THRESHOLD,
            imgsz=config.MODEL_IMAGE_SIZE,
            verbose=False
        )
    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}")

    # --- Parse detections — accumulate ALL boxes -----------------------------
    best_box        = None
    best_confidence = 0.0
    total_bbox_area = 0.0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Sum area across every detected pothole
            total_bbox_area += (x2 - x1) * (y2 - y1)

            if conf > best_confidence:
                best_confidence = conf
                best_box = box

    # --- No detection --------------------------------------------------------
    if best_box is None:
        return {
            config.RESPONSE_KEYS["pothole_detected"]: False,
            config.RESPONSE_KEYS["severity"]: "none",
            config.RESPONSE_KEYS["confidence"]: 0.0,
            config.RESPONSE_KEYS["severity_score"]: 0.0,
            config.RESPONSE_KEYS["bbox_width"]: 0,
            config.RESPONSE_KEYS["bbox_height"]: 0,
        }

    # --- Extract best bbox (kept for Java backward-compat) -------------------
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    bbox_w = int(round(x2 - x1))
    bbox_h = int(round(y2 - y1))

    # --- Severity (now uses summed area) -------------------------------------
    severity_label, severity_score = compute_severity(
        total_bbox_area, image_w, image_h
    )

    return {
        config.RESPONSE_KEYS["pothole_detected"]: True,
        config.RESPONSE_KEYS["severity"]: severity_label,
        config.RESPONSE_KEYS["confidence"]: round(best_confidence, 4),
        config.RESPONSE_KEYS["severity_score"]: severity_score,
        config.RESPONSE_KEYS["bbox_width"]: bbox_w,
        config.RESPONSE_KEYS["bbox_height"]: bbox_h,
    }
