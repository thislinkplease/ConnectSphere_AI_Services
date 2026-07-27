# ConnectSphere AI Moderation Service

AI Moderation Service is a FastAPI microservice used in ConnectSphere to check uploaded images and videos before they are stored or published. The service currently detects two types of unsafe content:

- Violent content
- Gambling-related content

The Node.js backend calls this service when users create posts or upload post media. If the AI service detects a violation, the backend blocks the upload and returns an error message to the client.

---

## 1. Main Features

- Image moderation using YOLO models
- Video moderation by extracting sampled frames
- Separate confidence thresholds for violence and gambling detection
- File validation by MIME type and file size
- JSON response showing whether the uploaded content is allowed
- Integration with the Express backend through `AI_MODERATION_URL`

---

## 2. Project Structure

AI service folder structure:

```txt
AI-Services/
├── main.py
├── models/
│   ├── violent_detection_model.pt
│   └── gambling_detection_model.pt
├── requirements.txt
├── README.md
└── venv/
```

---

## 3. Technologies Used

- Python
- FastAPI
- Uvicorn
- Ultralytics YOLO
- OpenCV
- Pillow
- NumPy

---

## 4. AI Models

The service loads two trained YOLO models:

```python
violent_model = YOLO("models/violent_detection_model.pt")
gambling_model = YOLO("models/gambling_detection_model.pt")
```

After loading, both models are warmed up with a dummy image to reduce delay on the first real request.

Current detection settings:

| Category | Model File | Blocked Label | Confidence Threshold |
|---|---|---|---|
| Violence | `violent_detection_model.pt` | `violent-action` | `0.80` |
| Gambling | `gambling_detection_model.pt` | `gambling-element` | `0.85` |

---

## 5. Supported File Types

### Images

- JPEG
- PNG
- WebP

### Videos

- MP4
- MOV / QuickTime
- AVI

Maximum file size:

```txt
50 MB
```

For videos, the service samples up to 12 frames and checks each selected frame with both YOLO models.

---

## 6. API Endpoints

### Health Check

```http
GET /
```

Example response:

```json
{
  "status": "ok",
  "models": ["violent_detection", "gambling_detection"]
}
```

---

### Moderate File

```http
POST /moderate
```

Request type:

```txt
multipart/form-data
```

Form field:

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Image or video file to moderate |

Example allowed response:

```json
{
  "allowed": true,
  "violations": [],
  "message": "Content is valid."
}
```

Example blocked response:

```json
{
  "allowed": false,
  "violations": [
    {
      "type": "violent",
      "label": "violent-action",
      "confidence": 0.91
    }
  ],
  "message": "Content violates the moderation policy and cannot be posted."
}
```

---

## 7. Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install fastapi uvicorn ultralytics pillow opencv-python numpy python-multipart
```

Optional: save dependencies to `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## 8. How to Run

From the AI service folder:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The service will run at:

```txt
http://localhost:8001
```

API documentation is available at:

```txt
http://localhost:8001/docs
```

---

## 9. Backend Integration

The Express backend connects to the AI service through:

```js
const AI_MODERATION_URL =
  process.env.AI_MODERATION_URL || "http://localhost:8001/moderate";
```

Recommended `.env` setting for the backend:

```env
AI_MODERATION_URL=http://localhost:8001/moderate
```

When a user creates a post with media, the backend sends each uploaded file to the AI service before inserting the post into the database. If the result is blocked, the backend returns status `400` with violation details. If the result is allowed, the backend continues uploading media to Supabase Storage and creates the post record.

---

## 10. Moderation Flow

```txt
User selects image/video
        ↓
React Native app sends post request to Node.js backend
        ↓
Backend receives media files using multer
        ↓
Backend sends each file to FastAPI AI Moderation Service
        ↓
FastAPI validates file type and size
        ↓
Image: YOLO checks image directly
Video: OpenCV extracts sampled frames, then YOLO checks each frame
        ↓
FastAPI returns allowed / blocked result
        ↓
Backend either blocks the post or stores it in Supabase
        ↓
Client shows success message or content blocked alert
```

---

## 11. Testing with cURL

Health check:

```bash
curl http://localhost:8001/
```

Moderate an image:

```bash
curl -X POST "http://localhost:8001/moderate" \
  -F "file=@test.jpg"
```

Moderate a video:

```bash
curl -X POST "http://localhost:8001/moderate" \
  -F "file=@test.mp4"
```

---

## 12. Common Errors

### `AI moderation service failed`

Possible causes:

- FastAPI service is not running
- Backend `AI_MODERATION_URL` is wrong
- Model files are missing from the `models/` folder
- Uploaded file type is not supported
- The AI service crashed while processing a large video

### `Unsupported file type`

The uploaded file MIME type is not included in the allowed image or video types.

### `Cannot read video`

OpenCV cannot read the uploaded video. Try converting the video to MP4.

### `Cannot read image`

The file is not a valid image or the image is corrupted.

---

## 13. Notes

- Thresholds can be adjusted to reduce false positives or false negatives.
- A higher threshold blocks fewer files but may miss some violations.
- A lower threshold catches more violations but may block valid content.

---

## 14. Future Improvements

- Improve dataset quality and retrain models for better accuracy
- Add more moderation categories such as nudity, hate symbols, or dangerous objects
- Store moderation logs for review
- Add asynchronous video processing for large files
- Add unit tests for image and video moderation
- Deploy the AI service separately from the main backend
