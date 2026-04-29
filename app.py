# =============================================================================
# app.py — Flask ML Microservice
# =============================================================================
#
# ENDPOINTS:
#   POST /detect          ← [JAVA-CONNECT] primary endpoint Java calls
#   GET  /health          ← [JAVA-CONNECT] Java health-check probe
#   GET  /model/info      ← [JAVA-CONNECT] Java admin/monitoring
#   POST /detect/batch    ← [JAVA-CONNECT] placeholder, not yet implemented
#
# HOW JAVA CALLS THIS SERVICE:
#   Java's MLDetectionService.java sends:
#       POST http://localhost:5000/detect
#       Headers:  Content-Type: multipart/form-data
#                 X-Internal-Secret: <INTERNAL_API_SECRET from config.py>
#       Body:     image=<file bytes>
#
# =============================================================================

from flask import Flask, request, jsonify
from PIL import Image
import io
import traceback
import os

import config
import detector

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Load model safely at startup (works with Gunicorn)
# -----------------------------------------------------------------------------

model_loaded = False

def ensure_model_loaded():
    global model_loaded
    if not model_loaded:
        try:
            detector.load_model()
            model_loaded = True
            print("[startup] Model loaded successfully")
        except Exception as e:
            print("[FATAL] Model loading failed:")
            traceback.print_exc()
            raise RuntimeError("Model loading failed")

# =============================================================================
# [JAVA-CONNECT] CORS — allow Java backend origin during local development
#   In production behind a reverse proxy this is not needed.
#   Java's base URL is set in config.JAVA_BACKEND_URL.
# =============================================================================
@app.route("/")
def home():
    return "Flask ML Service is running"
    
@app.after_request
def add_cors_headers(response):
    # Allow calls from Java backend and React frontend during dev
    allowed_origins = [
        config.JAVA_BACKEND_URL,   # e.g. http://localhost:8080
        "http://localhost:3000",   # React dev server
    ]
    origin = request.headers.get("Origin", "")
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"]  = origin
    response.headers["Access-Control-Allow-Methods"]     = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"]     = (
        "Content-Type, X-Internal-Secret"
    )
    return response


# =============================================================================
# [JAVA-CONNECT] Internal secret guard
#   Wrap any route with @require_internal_secret to enforce the shared secret.
#   Java must send header:  X-Internal-Secret: <value from config>
#
#   To DISABLE this check during development, set env:
#       INTERNAL_API_SECRET=skip
# =============================================================================
from functools import wraps

def require_internal_secret(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if config.INTERNAL_API_SECRET == "skip":
            return f(*args, **kwargs)                # dev bypass
        provided = request.headers.get("X-Internal-Secret", "")
        if provided != config.INTERNAL_API_SECRET:
            return jsonify({
                "error": "Unauthorized",
                "hint":  "Java must send X-Internal-Secret header"
            }), 401
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# Startup — load model once
# =============================================================================



# =============================================================================
# POST /detect
# =============================================================================
# [JAVA-CONNECT] PRIMARY ENDPOINT
#
# Java sends:
#   POST http://localhost:5000/detect
#   multipart/form-data
#   Field name: "image"   (must match config.IMAGE_FIELD_NAME)
#   Header:     X-Internal-Secret: <secret>
#
# Java reads response:
#   MLDetectionResult.java — fields must match JSON keys exactly:
#       pothole_detected  → boolean potholeDetected
#       severity          → String  severity
#       confidence        → double  confidence
#       severity_score    → double  severityScore
#       bbox_width        → int     bboxWidth
#       bbox_height       → int     bboxHeight
#
# Java maps JSON → Java using Jackson @JsonProperty annotations.
# =============================================================================
@app.route("/api/detect-pothole", methods=["POST"])
def detect_pothole_api():

    # 1. Validate image
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image provided",
            "data": None
        }), 400

    file = request.files["image"]

    # 2. Read image
    try:
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({
            "success": False,
            "message": "Invalid image",
            "data": str(e)
        }), 422

    # 3. Ensure model loaded
    try:
        ensure_model_loaded()
    except Exception:
        return jsonify({
            "success": False,
            "message": "Model not available",
            "data": None
        }), 503

    # 4. Run detection (reuse your detector logic)
    try:
        model = detector.get_model()

        results = model.predict(
            source=pil_image,
            conf=config.MODEL_CONFIDENCE_THRESHOLD,
            imgsz=config.MODEL_IMAGE_SIZE,
            verbose=False
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "message": "Detection failed",
            "data": str(e)
        }), 500

    # 5. Parse detections
    detections = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "id": len(detections) + 1,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 4)
            })

    pothole_count = len(detections)

    # 6. No pothole case
    if pothole_count == 0:
        return jsonify({
            "success": False,
            "message": "No pothole detected in the image",
            "data": {
                "hasPothole": False,
                "potholeCount": 0,
                "confidence": 0.0
            }
        }), 200   # better than 400

    # 7. Success
    return jsonify({
        "success": True,
        "message": f"Pothole detected! Found {pothole_count} pothole(s).",
        "data": {
            "hasPothole": True,
            "potholeCount": pothole_count,
            "confidence": max(d["confidence"] for d in detections),
            "detections": detections
        }
    }), 200
@app.route("/detect", methods=["POST"])
@require_internal_secret
def detect():
    # --- Validate image field ------------------------------------------------
    if config.IMAGE_FIELD_NAME not in request.files:
        return jsonify({
            "error": f"Missing field '{config.IMAGE_FIELD_NAME}' in multipart body",
            # [JAVA-CONNECT] Java checks this "error" key to surface user messages
        }), 400

    file = request.files[config.IMAGE_FIELD_NAME]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # --- Validate model loaded -----------------------------------------------
    # --- Ensure model is loaded ----------------------------------------------

    try:
        ensure_model_loaded()
    except RuntimeError:
        return jsonify({
            "error": "ML model not loaded",
            "hint":  "Place best.pt in model/ and restart Flask"
        }), 503

    # --- Read and convert image ----------------------------------------------
    try:
        image_bytes = file.read()
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({
            "error":   "Cannot read image",
            "detail":  str(e)
        }), 422

    # --- Run detection -------------------------------------------------------
    try:
        result = detector.detect_pothole(pil_image)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error":  "Detection failed",
            "detail": str(e)
        }), 500

    # --- Return exact format Java expects ------------------------------------
    # [JAVA-CONNECT] This is the EXACT JSON Java's MLDetectionResult.java parses.
    # Do NOT rename these keys without updating MLDetectionResult.java.
    return jsonify(result), 200


# =============================================================================
# GET /health
# =============================================================================
# [JAVA-CONNECT] Java Spring Boot calls this on startup and every 30s
#   to check if the ML service is up before forwarding report submissions.
#
#   Java usage in MLDetectionService.java:
#       GET http://localhost:5000/health
#       Expected: HTTP 200 + {"status": "ok", "model_loaded": true}
#
#   Java should refuse to accept pothole reports when model_loaded = false.
# =============================================================================
@app.route("/health", methods=["GET"])


def health():
    try:
        print("MODEL PATH:", config.MODEL_PATH)
        print("FILE EXISTS:", os.path.exists(config.MODEL_PATH))
        
        try:
            detector.get_model()
            model_loaded = True
        except Exception:
            model_loaded = False

        return jsonify({
            "status": "ok" if model_loaded else "model_not_loaded",
            "model_loaded": model_loaded
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Health check failed",
            "detail": str(e)
        }), 500

# =============================================================================
# GET /model/info
# =============================================================================
# [JAVA-CONNECT] Optional — Java admin dashboard can call this to display
#   ML model metadata in the authority panel.
#
#   Java usage: GET http://localhost:5000/model/info
#   Returns model config values for display purposes only.
# =============================================================================
@app.route("/model/info", methods=["GET"])
@require_internal_secret
def model_info():
    return jsonify({
        "model_path":             config.MODEL_PATH,
        "confidence_threshold":   config.MODEL_CONFIDENCE_THRESHOLD,
        "image_size":             config.MODEL_IMAGE_SIZE,
        "severity_thresholds": {
            "low_max":    config.SEVERITY_LOW_MAX,
            "medium_max": config.SEVERITY_MEDIUM_MAX,
        },
        "expected_input_field":   config.IMAGE_FIELD_NAME,
        # [JAVA-CONNECT] Java admin panel can render these as read-only config
    }), 200


# =============================================================================
# POST /detect/batch   — PLACEHOLDER
# =============================================================================
# [JAVA-CONNECT] Future endpoint for bulk image processing.
#   Java will call this when processing multiple reports at once
#   (e.g. authority bulk upload, drone footage frames).
#
#   NOT implemented yet. Returns 501 so Java knows it's intentionally absent.
#
#   When implemented:
#       Input:  multipart with fields image_0, image_1, ... image_N
#       Output: {"results": [ <detect response>, ... ]}
# =============================================================================
@app.route("/detect/batch", methods=["POST"])
@require_internal_secret
def detect_batch():
    return jsonify({
        "error":  "Not implemented",
        "hint":   "Batch detection is a planned feature",
        # [JAVA-CONNECT] Java should check for 501 and fall back to single /detect
    }), 501


# =============================================================================
# OPTIONS — preflight for CORS
# =============================================================================
@app.route("/detect",       methods=["OPTIONS"])
@app.route("/detect/batch", methods=["OPTIONS"])
@app.route("/model/info",   methods=["OPTIONS"])
def options_handler():
    return "", 204


# =============================================================================
# Global error handlers
# =============================================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# =============================================================================
# Entry point (only for local running, NOT used by Gunicorn)
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

