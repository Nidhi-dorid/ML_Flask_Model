# =============================================================================
# detector.py — YOLOv8 inference engine + severity scoring
# =============================================================================
# Loaded once at startup. Flask calls detect_pothole() per request.
# No Java integration points here — this is pure ML logic.
# =============================================================================

import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import config

# -----------------------------------------------------------------------------
# Model singleton — loaded once when Flask starts
# -----------------------------------------------------------------------------
_model = None

def load_model():
    """
    Load YOLOv8 model from disk.
    Called once at Flask startup via app.py.
    Crashes loudly if model file is missing — intentional, fail fast.
    """
    global _model
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{config.MODEL_PATH}'.\n"
            f"Train YOLOv8 and copy best.pt to that path.\n"
            f"See train/train.py for training script."
        )
    _model = YOLO(config.MODEL_PATH)
    print(f"[detector] Model loaded from {config.MODEL_PATH}")
    return _model


def get_model():
    """Return the loaded model, raising if not initialized yet."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


# -----------------------------------------------------------------------------
# Severity scoring
# -----------------------------------------------------------------------------

def compute_severity(bbox_width: int, bbox_height: int,
                     image_width: int, image_height: int) -> tuple[str, float]:
    """
    Derive severity label and 0–1 score from bounding box size.

    Logic:
        severity_score = bbox_area / image_area   (clamped to 1.0)
        score < SEVERITY_LOW_MAX    → "low"
        score < SEVERITY_MEDIUM_MAX → "medium"
        score ≥ SEVERITY_MEDIUM_MAX → "high"

    Returns:
        (severity_label: str, severity_score: float)
    """
    image_area = image_width * image_height
    if image_area == 0:
        return "low", 0.0

    bbox_area    = bbox_width * bbox_height
    raw_score    = bbox_area / image_area
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
    """
    Run YOLOv8 inference on a PIL image.

    Args:
        pil_image: PIL.Image.Image in RGB mode

    Returns dict with EXACT keys Java expects (see config.RESPONSE_KEYS):
        {
            "pothole_detected": bool,
            "severity":         str   ("low" | "medium" | "high"),
            "confidence":       float (0.0–1.0),
            "severity_score":   float (0.0–1.0),
            "bbox_width":       int,
            "bbox_height":      int
        }

    If no pothole detected:
        pothole_detected = False
        severity         = "none"
        confidence       = 0.0
        severity_score   = 0.0
        bbox_width       = 0
        bbox_height      = 0
    """
    model        = get_model()
    image_w, image_h = pil_image.size

    # --- Run inference -------------------------------------------------------
    results = model.predict(
        source    = pil_image,
        conf      = config.MODEL_CONFIDENCE_THRESHOLD,
        imgsz     = config.MODEL_IMAGE_SIZE,
        verbose   = False
    )

    # --- Parse detections ----------------------------------------------------
    best_box        = None
    best_confidence = 0.0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_confidence:
                best_confidence = conf
                best_box        = box

    # --- No detection --------------------------------------------------------
    if best_box is None:
        return {
            config.RESPONSE_KEYS["pothole_detected"]: False,
            config.RESPONSE_KEYS["severity"]:         "none",
            config.RESPONSE_KEYS["confidence"]:       0.0,
            config.RESPONSE_KEYS["severity_score"]:   0.0,
            config.RESPONSE_KEYS["bbox_width"]:       0,
            config.RESPONSE_KEYS["bbox_height"]:      0,
        }

    # --- Extract bbox --------------------------------------------------------
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    bbox_w = int(round(x2 - x1))
    bbox_h = int(round(y2 - y1))

    # --- Severity ------------------------------------------------------------
    severity_label, severity_score = compute_severity(
        bbox_w, bbox_h, image_w, image_h
    )

    return {
        config.RESPONSE_KEYS["pothole_detected"]: True,
        config.RESPONSE_KEYS["severity"]:         severity_label,
        config.RESPONSE_KEYS["confidence"]:       round(best_confidence, 4),
        config.RESPONSE_KEYS["severity_score"]:   severity_score,
        config.RESPONSE_KEYS["bbox_width"]:        bbox_w,
        config.RESPONSE_KEYS["bbox_height"]:       bbox_h,
    }
