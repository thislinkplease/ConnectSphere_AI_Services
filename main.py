from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import io

app = FastAPI(title="Content Moderation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

violent_model = YOLO("models/violent_detection_model.pt")
gambling_model = YOLO("models/gambling_detection_model.pt")

_dummy = Image.new("RGB", (640, 640), color="white")
violent_model(_dummy, imgsz=640, verbose=False)
gambling_model(_dummy, imgsz=640, verbose=False)
print("AI moderation models warmed up")

VIOLENT_CONFIDENCE_THRESHOLD = 0.8
GAMBLING_CONFIDENCE_THRESHOLD = 0.85

BLOCKED_VIOLENT_LABELS = {"violent-action"}
BLOCKED_GAMBLING_LABELS = {"gambling-element"}

MAX_VIDEO_FRAMES = 12
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}
MAX_FILE_SIZE_MB = 50


# ────────────────────────────────────── Helpers ──────────────────────────────────────

def validate_file(file: UploadFile, data: bytes):
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File is too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")

    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, f"Unsupported file type: {content_type}")

    return content_type


def predict_image(img: Image.Image) -> list[dict]:
    v_results = violent_model(img, imgsz=640, verbose=False)[0]
    g_results = gambling_model(img, imgsz=640, verbose=False)[0]

    violations = []

    for box in v_results.boxes:
        conf = float(box.conf)
        label = str(violent_model.names[int(box.cls)]).strip().lower()

        print(f"[VIOLENT MODEL] label={label}, conf={conf:.3f}")

        if label in BLOCKED_VIOLENT_LABELS and conf >= VIOLENT_CONFIDENCE_THRESHOLD:
            violations.append({
                "type": "violent",
                "label": label,
                "confidence": round(conf, 3),
            })

    for box in g_results.boxes:
        conf = float(box.conf)
        label = str(gambling_model.names[int(box.cls)]).strip().lower()

        print(f"[GAMBLING MODEL] label={label}, conf={conf:.3f}")

        if label in BLOCKED_GAMBLING_LABELS and conf >= GAMBLING_CONFIDENCE_THRESHOLD:
            violations.append({
                "type": "gambling",
                "label": label,
                "confidence": round(conf, 3),
            })

    return violations


def extract_frames(video_bytes: bytes, max_frames: int = MAX_VIDEO_FRAMES) -> list[Image.Image]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            raise HTTPException(400, "Cannot read video.")

        indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))

        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames


# ────────────────────────────────────── Endpoints ──────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "models": ["violent_detection", "gambling_detection"]}


@app.post("/moderate")
async def moderate(file: UploadFile = File(...)):
    data = await file.read()
    content_type = validate_file(file, data)
    is_video = content_type in ALLOWED_VIDEO_TYPES

    all_violations = []

    if not is_video:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Cannot read image.")

        all_violations = predict_image(img)
    else:
        frames = extract_frames(data)

        if not frames:
            raise HTTPException(400, "Cannot extract video frames.")

        seen = set()

        for idx, frame in enumerate(frames):
            frame_violations = predict_image(frame)

            for violation in frame_violations:
                key = (violation["type"], violation["label"])
                if key not in seen:
                    seen.add(key)
                    all_violations.append(violation)

            if all_violations:
                print(f"[VIDEO MODERATION] Blocked at frame {idx + 1}/{len(frames)}")
                break

    is_allowed = len(all_violations) == 0

    if not is_allowed:
        return {
            "allowed": False,
            "violations": all_violations,
            "message": "Content violates the moderation policy and cannot be posted.",
        }

    return {
        "allowed": True,
        "violations": [],
        "message": "Content is valid.",
    }
