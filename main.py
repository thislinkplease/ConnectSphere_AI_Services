from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import asyncio
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

# Load 1 lần khi server khởi động
violent_model    = YOLO("models/violent_detection_model.pt")
gambling_model   = YOLO("models/gambling_detection_model.pt")

CONFIDENCE_THRESHOLD = 0.6   # >= 60% → vi phạm
MAX_VIDEO_FRAMES     = 10    # sample tối đa 10 frames mỗi video
ALLOWED_IMAGE_TYPES  = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES  = {"video/mp4", "video/quicktime", "video/x-msvideo"}
MAX_FILE_SIZE_MB     = 50


# ── Helpers ──────────────────────────────────────────────────────────────────

def validate_file(file: UploadFile, data: bytes):
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File quá lớn, tối đa {MAX_FILE_SIZE_MB}MB")
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, f"Định dạng không hỗ trợ: {content_type}")
    return content_type


def predict_image(img: Image.Image) -> dict:
    """Chạy cả 2 model trên 1 ảnh PIL, trả về kết quả gộp."""
    img_resized = img.resize((640, 640))

    v_results = violent_model(img_resized, verbose=False)[0]
    g_results = gambling_model(img_resized, verbose=False)[0]

    violations = []

    for box in v_results.boxes:
        conf = float(box.conf)
        if conf >= CONFIDENCE_THRESHOLD:
            violations.append({
                "type":       "violent",
                "label":      violent_model.names[int(box.cls)],
                "confidence": round(conf, 3),
            })

    for box in g_results.boxes:
        conf = float(box.conf)
        if conf >= CONFIDENCE_THRESHOLD:
            violations.append({
                "type":       "gambling",
                "label":      gambling_model.names[int(box.cls)],
                "confidence": round(conf, 3),
            })

    return violations


def extract_frames(video_bytes: bytes, max_frames: int = MAX_VIDEO_FRAMES) -> list[Image.Image]:
    """Giải mã video từ bytes, lấy đều N frames."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise HTTPException(400, "Không đọc được video")

        # Lấy đều tối đa max_frames frames
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "models": ["violent_detection", "gambling_detection"]}


@app.post("/moderate")
async def moderate(file: UploadFile = File(...)):
    """
    Endpoint duy nhất cho cả ảnh lẫn video.
    Response:
      - allowed: true/false
      - violations: danh sách vi phạm (rỗng nếu sạch)
    """
    data         = await file.read()
    content_type = validate_file(file, data)
    is_video     = content_type in ALLOWED_VIDEO_TYPES

    all_violations = []

    if not is_video:
        # ── Ảnh ──
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Không đọc được ảnh")
        all_violations = predict_image(img)

    else:
        # ── Video: extract frames rồi predict từng frame ──
        frames = extract_frames(data)
        if not frames:
            raise HTTPException(400, "Không extract được frames")

        # Chạy song song các frame
        loop    = asyncio.get_event_loop()
        results = await asyncio.gather(*[
            loop.run_in_executor(None, predict_image, frame)
            for frame in frames
        ])

        # Gộp tất cả vi phạm, loại trùng lặp
        seen = set()
        for frame_violations in results:
            for v in frame_violations:
                key = (v["type"], v["label"])
                if key not in seen:
                    seen.add(key)
                    all_violations.append(v)

    is_allowed = len(all_violations) == 0

    if not is_allowed:
        return {
            "allowed":    False,
            "violations": all_violations,
            "message":    "Nội dung vi phạm, không thể đăng bài",
        }

    return {
        "allowed":    True,
        "violations": [],
        "message":    "Nội dung hợp lệ",
    }