# =============================================================================
# config.py — Central configuration for Flask ML Service
# =============================================================================
# JAVA INTEGRATION NOTE:
#   Every value marked [JAVA-CONNECT] is a placeholder that the Java backend
#   either sets via environment variable OR calls directly.
#   Search "JAVA-CONNECT" across all files to find every integration point.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# Base directory (ensures paths work in local + Render)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# -----------------------------------------------------------------------------
# Flask server config
# -----------------------------------------------------------------------------
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# -----------------------------------------------------------------------------
# [JAVA-CONNECT] Java Spring Boot base URL
#   - Java calls Flask at POST /detect
#   - This URL is used only if Flask ever needs to call back to Java (future)
#   - In Docker: set JAVA_BACKEND_URL=http://java-backend:8080
#   - Locally:   set JAVA_BACKEND_URL=http://localhost:8080
# -----------------------------------------------------------------------------
JAVA_BACKEND_URL = os.getenv("JAVA_BACKEND_URL", "http://localhost:8080")

# -----------------------------------------------------------------------------
# [JAVA-CONNECT] Shared API secret between Java and Flask
#   - Java must send this in the X-Internal-Secret header on every /detect call
#   - Prevents random external callers from hitting the ML service directly
#   - Java sets this in its application.properties as ml.service.secret
#   - CHANGE THIS VALUE in production — never commit the real secret
# -----------------------------------------------------------------------------
INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET", "dev-secret-change-in-prod")

# -----------------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "model", "best.pt")   # ✅ IMPORTANT FIX
)
MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.65"))
MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", "640"))

# -----------------------------------------------------------------------------
# Severity thresholds
#   Severity is computed from bbox area as fraction of total image area.
#   Tune these after training is done.
# -----------------------------------------------------------------------------
# NEW (matches your notebook's calibrated thresholds)
SEVERITY_LOW_MAX    = float(os.getenv("SEVERITY_LOW_MAX",    "0.02"))
SEVERITY_MEDIUM_MAX = float(os.getenv("SEVERITY_MEDIUM_MAX", "0.08"))
                                                                          # ≥ 15% → high

# -----------------------------------------------------------------------------
# [JAVA-CONNECT] Expected request field names
#   Java sends a multipart/form-data POST to /detect.
#   The image file must be under this field name.
#   Must match: MLDetectionService.java → body.add("<THIS_VALUE>", ...)
# -----------------------------------------------------------------------------
IMAGE_FIELD_NAME = "image"

# -----------------------------------------------------------------------------
# [JAVA-CONNECT] Response field names
#   The JSON keys Flask returns.
#   Must match: MLDetectionResult.java field names (camelCase mapped by Jackson)
# -----------------------------------------------------------------------------
RESPONSE_KEYS = {
    "pothole_detected": "pothole_detected",   # boolean
    "severity":         "severity",           # "low" | "medium" | "high"
    "confidence":       "confidence",         # float 0.0–1.0
    "severity_score":   "severity_score",     # float 0.0–1.0
    "bbox_width":       "bbox_width",         # int pixels
    "bbox_height":      "bbox_height",        # int pixels
}
