# מודלים מומלצים ל-PrivacyFlow מ-Hugging Face

## סיכום מהיר

| תפקיד | מודל מומלץ | גודל | קישור |
|-------|-----------|------|-------|
| **זיהוי אנשים** | YOLOv8n-person | ~12MB | [deepghs/yolo-person](https://huggingface.co/deepghs/yolo-person) |
| **זיהוי פנים** | YOLOv8n-face | ~12MB | [deepghs/yolo-face](https://huggingface.co/deepghs/yolo-face) |
| **Pose Estimation** | MoveNet Lightning | ~9MB | [Xenova/movenet-singlepose-lightning](https://huggingface.co/Xenova/movenet-singlepose-lightning) |
| **Face Embedding** | ArcFace (להתאמה) | ~250MB | [garavv/arcface-onnx](https://huggingface.co/garavv/arcface-onnx) |

---

## 1. 🚶 Person Detection - זיהוי אנשים

### מומלץ: `deepghs/yolo-person` (YOLOv8n)

**למה?**
- מאומן ספציפית לזהות אנשים בלבד (לא 80 קטגוריות כמו COCO)
- קל מאוד - רק 12MB
- ONNX מוכן
- מהיר מספיק ל-Raspberry Pi

**הורדה:**
```python
from huggingface_hub import hf_hub_download

# הורדת המודל
model_path = hf_hub_download(
    repo_id="deepghs/yolo-person",
    filename="yolov8n-person/model.onnx",
    local_dir="models"
)
```

**שימוש:**
```python
import onnxruntime as ort
import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path="models/yolov8n-person/model.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (640, 640)  # YOLOv8 standard
    
    def preprocess(self, frame):
        img = cv2.resize(frame, self.input_shape)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        return img[np.newaxis, ...]  # Add batch dimension
    
    def detect(self, frame, confidence_threshold=0.5):
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        # Parse YOLO output format...
        return self._parse_detections(outputs, confidence_threshold)
```

### אלטרנטיבות:
| מודל | גודל | יתרון |
|------|------|-------|
| `Ultralytics/YOLOv8` (yolov8n) | 6MB | הכי קל, אבל מזהה 80 קטגוריות |
| `YOLOX-nano` | ~4MB | קטן מאוד, anchor-free |
| `Ultralytics/YOLO11` (yolo11n) | ~5MB | הכי חדש, ביצועים טובים |

---

## 2. 😊 Face Detection - זיהוי פנים

### מומלץ: `deepghs/yolo-face` (YOLOv8n)

**למה?**
- מאומן ספציפית על פנים
- קל - 12MB
- ONNX מוכן
- מספיק טוב לחתוך פנים ל-32×32

**הורדה:**
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="deepghs/yolo-face",
    filename="yolov8n-face/model.onnx",
    local_dir="models"
)
```

### אלטרנטיבה: `AdamCodd/YOLOv11n-face-detection`
- YOLOv11 nano - חדש יותר
- Easy Val AP: 94.2%

---

## 3. 🏃 Pose Estimation - הערכת תנוחה

### מומלץ: `Xenova/movenet-singlepose-lightning`

**למה?**
- **קל מאוד** - ~9MB
- מהיר - <7ms על מובייל
- 17 נקודות מפתח (COCO format)
- מושלם ל-Raspberry Pi

**הורדה:**
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Xenova/movenet-singlepose-lightning",
    filename="onnx/model.onnx",
    local_dir="models"
)
```

**שימוש:**
```python
import onnxruntime as ort
import numpy as np
import cv2

class PoseEstimator:
    def __init__(self, model_path="models/movenet-singlepose-lightning/onnx/model.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_size = (192, 192)  # MoveNet Lightning input
        
    def estimate(self, person_crop):
        # Preprocess
        img = cv2.resize(person_crop, self.input_size)
        img = img.astype(np.int32)  # MoveNet expects int32
        img = img[np.newaxis, ...]  # Add batch: (1, 192, 192, 3)
        
        # Run inference
        outputs = self.session.run(None, {"input": img})
        keypoints = outputs[0][0, 0]  # Shape: (17, 3) - y, x, confidence
        
        return keypoints

# 17 Keypoints (COCO format):
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

### אלטרנטיבות:
| מודל | גודל | יתרון |
|------|------|-------|
| `Xenova/movenet-multipose-lightning` | ~19MB | מזהה מספר אנשים בו-זמנית |
| `Xenova/movenet-singlepose-thunder` | ~13MB | יותר מדויק, קצת יותר איטי |
| `amd/movenet` (int8) | ~7.7MB | Quantized - מהיר יותר |
| `Ultralytics/YOLOv8` (pose) | ~7MB | YOLO עם pose מובנה |

---

## 4. 🔐 Face Embedding - זיהוי פנים (לפרויקט שלך)

### ⚠️ הערה חשובה לפרטיות

המודלים הסטנדרטיים (ArcFace) מקבלים תמונות **112×112** ומפיקים embeddings של **512 dimensions**.
**אתה רוצה 32×32** - זה בכוונה כדי לשמור על פרטיות!

### אפשרות 1: שימוש ב-ArcFace עם Downscale (פשוט אבל פחות פרטי)

```python
from huggingface_hub import hf_hub_download

# הורדת ArcFace
model_path = hf_hub_download(
    repo_id="garavv/arcface-onnx",
    filename="arcface.onnx",
    local_dir="models"
)
```

```python
import cv2
import numpy as np
import onnxruntime as ort

class FaceEncoder:
    def __init__(self, model_path="models/arcface.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        # ArcFace expects 112x112
        self.input_size = (112, 112)
        
    def encode(self, face_image_32x32):
        """
        מקבל תמונת פנים 32×32 ומחזיר embedding.
        
        לפרטיות: אנחנו עובדים עם 32×32 מקור,
        גם אם המודל מצפה ל-112×112
        """
        # Scale up from 32×32 to 112×112 (intentionally blurry!)
        face_upscaled = cv2.resize(
            face_image_32x32, 
            self.input_size,
            interpolation=cv2.INTER_LINEAR  # Keep it blurry
        )
        
        # Normalize
        face = (face_upscaled.astype(np.float32) - 127.5) / 128.0
        face = face.transpose(2, 0, 1)  # HWC -> CHW
        face = face[np.newaxis, ...]  # Add batch
        
        # Get embedding
        embedding = self.session.run(None, {self.input_name: face})[0][0]
        
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        
        # Optional: Reduce dimensions for more privacy
        # Take only first 64 dimensions instead of 512
        return embedding[:64]
```

### אפשרות 2: בניית מודל פשוט משלך (מומלץ לפרטיות!)

מכיוון שאתה רוצה **בכוונה** זיהוי לא מדויק, אתה יכול לבנות encoder פשוט:

```python
import numpy as np
import cv2

class SimplePrivacyFaceEncoder:
    """
    Face encoder שעובד על 32×32 פיקסלים בלבד.
    מחזיר embedding של 64 dimensions.
    
    זה בכוונה לא מדויק - זה ה-privacy feature!
    """
    
    def __init__(self):
        self.output_dim = 64
        self.input_size = (32, 32)
    
    def encode(self, face_32x32):
        """
        יוצר embedding מפשט מתמונת פנים 32×32.
        משתמש בשילוב של:
        1. Color histogram
        2. Gradient features (HOG-like)
        3. Spatial averages
        """
        # Ensure correct size
        if face_32x32.shape[:2] != self.input_size:
            face_32x32 = cv2.resize(face_32x32, self.input_size)
        
        features = []
        
        # 1. Color features (18 dims)
        for channel in cv2.split(face_32x32):
            hist = cv2.calcHist([channel], [0], None, [6], [0, 256])
            features.extend(hist.flatten() / hist.sum())
        
        # 2. Gradient features (32 dims)
        gray = cv2.cvtColor(face_32x32, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        
        # Divide into 4x4 grid and compute mean gradient
        for i in range(4):
            for j in range(4):
                cell = mag[i*8:(i+1)*8, j*8:(j+1)*8]
                features.append(np.mean(cell))
                features.append(np.std(cell))
        
        # 3. Spatial average features (14 dims to reach 64)
        for i in range(2):
            for j in range(2):
                for c in range(3):
                    cell = face_32x32[i*16:(i+1)*16, j*16:(j+1)*16, c]
                    features.append(np.mean(cell) / 255.0)
        features.extend([0, 0])  # Padding to 64
        
        embedding = np.array(features[:self.output_dim], dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
```

---

## 5. 📥 סקריפט להורדת כל המודלים

```python
#!/usr/bin/env python3
"""
Download all required models for PrivacyFlow.
"""

from huggingface_hub import hf_hub_download
import os

MODELS = {
    "person_detector": {
        "repo": "deepghs/yolo-person",
        "file": "yolov8n-person/model.onnx",
        "local": "models/person_detector.onnx"
    },
    "face_detector": {
        "repo": "deepghs/yolo-face",
        "file": "yolov8n-face/model.onnx",
        "local": "models/face_detector.onnx"
    },
    "pose_estimator": {
        "repo": "Xenova/movenet-singlepose-lightning",
        "file": "onnx/model.onnx",
        "local": "models/pose_estimator.onnx"
    },
    # Optional: ArcFace for face embedding
    # "face_encoder": {
    #     "repo": "garavv/arcface-onnx",
    #     "file": "arcface.onnx",
    #     "local": "models/face_encoder.onnx"
    # }
}

def download_models():
    os.makedirs("models", exist_ok=True)
    
    for name, config in MODELS.items():
        print(f"Downloading {name}...")
        try:
            path = hf_hub_download(
                repo_id=config["repo"],
                filename=config["file"]
            )
            # Copy or symlink to local path
            import shutil
            shutil.copy(path, config["local"])
            print(f"  ✓ Saved to {config['local']}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

if __name__ == "__main__":
    download_models()
```

---

## 6. 📊 טבלת השוואה מלאה

| מודל | משימה | גודל | Input | Output | FPS (RPi4) | קישור |
|------|-------|------|-------|--------|------------|-------|
| YOLOv8n-person | Person Detection | 12MB | 640×640 | Bboxes | ~5-10 | [HF](https://huggingface.co/deepghs/yolo-person) |
| YOLOv8n-face | Face Detection | 12MB | 640×640 | Bboxes | ~5-10 | [HF](https://huggingface.co/deepghs/yolo-face) |
| MoveNet Lightning | Pose | 9MB | 192×192 | 17 keypoints | ~15-20 | [HF](https://huggingface.co/Xenova/movenet-singlepose-lightning) |
| MoveNet Thunder | Pose (accurate) | 13MB | 256×256 | 17 keypoints | ~8-12 | [HF](https://huggingface.co/Xenova/movenet-singlepose-thunder) |
| ArcFace | Face Embedding | 250MB | 112×112 | 512-dim | ~3-5 | [HF](https://huggingface.co/garavv/arcface-onnx) |
| YOLOv8n-pose | Detection+Pose | 7MB | 640×640 | Bboxes+Keypoints | ~5-8 | [Ultralytics](https://huggingface.co/Ultralytics/YOLOv8) |

---

## 7. 🎯 המלצה הסופית לפרויקט שלך

בהתחשב ב:
- חומרה: ~$150 (Raspberry Pi 4)
- דרישת פרטיות: 32×32 פנים
- מטרה: ספירת נוסעים, לא זיהוי אישי

### Stack מומלץ:

```
1. Person Detection:     deepghs/yolo-person (YOLOv8n)     - 12MB
2. Pose Estimation:      Xenova/movenet-singlepose-lightning - 9MB  
3. Face Embedding:       SimplePrivacyFaceEncoder (מותאם אישית) - 0MB
4. Appearance:           Color histogram + clothing detector (מותאם אישית)

סה"כ: ~21MB models + custom code
```

### למה לא ArcFace/InsightFace?
- הם מדויקים מדי - זה **לא** מה שאתה רוצה!
- מקבלים 112×112 - לא 32×32
- מייצרים 512-dim embeddings - overkill לפרויקט פרטיות
- כבדים מדי ל-Raspberry Pi

---

## 8. 🔧 התקנה מהירה

```bash
# התקנת dependencies
pip install huggingface_hub onnxruntime opencv-python numpy

# הורדת מודלים
python scripts/download_models.py

# בדיקה
python -c "import onnxruntime; print('ONNX Runtime OK')"
```

---

## קישורים שימושיים

- [Hugging Face ONNX Models](https://huggingface.co/models?library=onnx)
- [ONNX Model Zoo (archived)](https://huggingface.co/onnxmodelzoo)
- [Ultralytics YOLO](https://huggingface.co/Ultralytics)
- [InsightFace](https://huggingface.co/deepghs/insightface)
- [MoveNet](https://huggingface.co/Xenova/movenet-singlepose-lightning)
