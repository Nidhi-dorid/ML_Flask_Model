# Flask ML Service — Java Integration Reference

This document is written for the Java Spring Boot developer.
Every `[JAVA-CONNECT]` comment in the codebase points back to a section here.

---

## 1. Endpoints

| Method | URL | Auth | Purpose |
|--------|-----|------|---------|
| POST | `/detect` | X-Internal-Secret | Run YOLOv8 on image → JSON result |
| GET | `/health` | none | Spring Boot health probe |
| GET | `/model/info` | X-Internal-Secret | Admin: model config |
| POST | `/detect/batch` | X-Internal-Secret | NOT YET IMPLEMENTED (501) |

---

## 2. POST /detect — request format

Java sends a `multipart/form-data` POST:

```
POST http://localhost:5000/detect
Headers:
    Content-Type:      multipart/form-data
    X-Internal-Secret: dev-secret-change-in-prod
Body:
    image = <file bytes>   ← field name must be "image"
```

Java code (MLDetectionService.java):
```java
MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("image", new MultipartInputStreamFileResource(
    image.getInputStream(), image.getOriginalFilename()
));
HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.MULTIPART_FORM_DATA);
headers.set("X-Internal-Secret", mlServiceSecret);  // from application.properties
```

---

## 3. POST /detect — response format (EXACT)

```json
{
    "pothole_detected": true,
    "severity":         "high",
    "confidence":       0.91,
    "severity_score":   0.85,
    "bbox_width":       150,
    "bbox_height":      120
}
```

When no pothole found:
```json
{
    "pothole_detected": false,
    "severity":         "none",
    "confidence":       0.0,
    "severity_score":   0.0,
    "bbox_width":       0,
    "bbox_height":      0
}
```

Java DTO (MLDetectionResult.java):
```java
public class MLDetectionResult {
    @JsonProperty("pothole_detected") private boolean potholeDetected;
    @JsonProperty("severity")         private String  severity;
    @JsonProperty("confidence")       private double  confidence;
    @JsonProperty("severity_score")   private double  severityScore;
    @JsonProperty("bbox_width")       private int     bboxWidth;
    @JsonProperty("bbox_height")      private int     bboxHeight;
}
```

---

## 4. GET /health — response format

```json
{ "status": "ok", "model_loaded": true, "model_path": "model/best.pt" }
```

Java should call this on startup and gate report submissions on `model_loaded == true`.

---

## 5. Shared secret

Both sides must share the same secret string.

Flask side (environment variable):
```
INTERNAL_API_SECRET=dev-secret-change-in-prod
```

Java side (application.properties):
```properties
ml.service.secret=dev-secret-change-in-prod
ml.service.url=http://localhost:5000
```

To disable during development, set Flask env:
```
INTERNAL_API_SECRET=skip
```

---

## 6. Error responses

Flask always returns errors as:
```json
{ "error": "human readable message", "detail": "optional technical detail" }
```

HTTP status codes:
- `400` — missing/invalid image field
- `401` — wrong X-Internal-Secret
- `422` — image file unreadable
- `500` — detection crashed (check Flask logs)
- `503` — model not loaded yet

---

## 7. Priority score calculation (Java side)

Flask returns `severity_score` (0–1) and `confidence` (0–1).
Java combines these with traffic/zone data into a `priority_score`:

```java
// Suggested formula in Java's PriorityService.java
int priorityScore = (int)(
    (mlResult.getSeverityScore() * 0.5 +
     mlResult.getConfidence()    * 0.3 +
     zoneWeightFactor            * 0.2) * 100
);
```

`zoneWeightFactor` = 1.0 if near hospital/school, 0.5 otherwise.

---

## 8. Local run (no Docker)

```bash
cd flask-ml-service
pip install -r requirements.txt

# Dev mode (no secret check)
INTERNAL_API_SECRET=skip python app.py

# Test
curl -X POST http://localhost:5000/detect \
     -F "image=@test_pothole.jpg"

curl http://localhost:5000/health
```
