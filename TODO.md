# PrivacyFlow - Implementation Checklist

## ✅ Already Implemented

### Core Modules
- [x] `src/privacy_token.py` - Token generation with daily salt rotation
- [x] `src/data_structures.py` - DetectionEvent, DailyTrajectory, FeatureVector
- [x] `src/trajectory.py` - TrajectoryBuilder with indexed lookups
- [x] `src/database.py` - EphemeralDatabase (SQLite in-memory)
- [x] `src/analytics.py` - Analytics, OccupancyMonitor, PlanningReport
- [x] `src/station_graph.py` - Graph-based route analysis (DFS/BFS)
- [x] `src/scheduler.py` - Daily purge scheduler
- [x] `src/detector.py` - PersonDetector (⚠️ placeholder implementation)
- [x] `src/api.py` - REST API endpoints
- [x] `src/privacyflow.py` - Main controller

### Tests & Examples
- [x] `tests/test_privacy.py` - Privacy guarantee tests
- [x] `tests/test_performance.py` - Performance benchmarks
- [x] `examples/basic_usage.py` - Basic usage examples
- [x] `examples/graph_analysis.py` - Graph analysis examples

### Configuration & Documentation
- [x] `config.yaml` - Configuration file
- [x] `requirements.txt` - Dependencies
- [x] `README.md` - Documentation
- [x] `.gitignore` - Git ignore rules
- [x] `LICENSE` - MIT License

---

## ❌ Missing - Must Implement

### 1. 🎯 ML Models (Critical)
The system needs actual ONNX models. Current `detector.py` has placeholders.

```
models/
├── face_encoder_32x32.onnx    # Custom face encoder for 32×32 input
├── pose_lite.onnx             # Lightweight pose estimation (MoveNet/MediaPipe)
├── appearance_encoder.onnx    # Appearance feature extractor
└── person_detector.onnx       # YOLO-Nano or similar for person detection
```

**Options:**
1. **Use Pre-trained Models** (Recommended for MVP):
   - Person Detection: YOLOv8-nano, MobileNet-SSD
   - Pose: MoveNet Lightning, MediaPipe Pose
   - Face: Train custom or use downscaled FaceNet

2. **Train Custom Models**:
   - Need training scripts in `scripts/train_*.py`

### 2. 🎥 Camera Integration (Critical)
Create `src/camera.py`:

```python
# src/camera.py - Need to implement
class CameraCapture:
    """Real-time camera capture for Raspberry Pi / USB cameras."""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        pass
    
    def start(self) -> None:
        """Start camera capture thread."""
        pass
    
    def get_frame(self) -> np.ndarray:
        """Get latest frame."""
        pass
    
    def stop(self) -> None:
        """Stop camera capture."""
        pass
```

### 3. 🚀 Main Entry Point (Critical)
Create `main.py`:

```python
# main.py - Need to implement
#!/usr/bin/env python3
"""Main entry point for PrivacyFlow system."""

import argparse
from src.privacyflow import PrivacyFlow
from src.camera import CameraCapture

def main():
    parser = argparse.ArgumentParser(description='PrivacyFlow')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--station-id', required=True)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    
    pf = PrivacyFlow(args.config)
    camera = CameraCapture(args.camera)
    
    pf.start()
    camera.start()
    
    try:
        while True:
            frame = camera.get_frame()
            pf.process_frame(frame, args.station_id, f"cam_{args.camera}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        pf.stop()

if __name__ == "__main__":
    main()
```

### 4. 🔧 Real Detection Implementation (Critical)
Update `src/detector.py` to use actual models:

```python
# In PersonDetector.detect() - currently returns []
# Need to implement actual YOLO/SSD inference
def detect(self, frame: np.ndarray) -> List[Detection]:
    # 1. Run person detection model
    # 2. For each person, extract face at 32×32
    # 3. Run pose estimation
    # 4. Return Detection objects
    pass
```

---

## 📦 Deployment Files Needed

### 5. Dockerfile
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py", "--config", "config.yaml", "--station-id", "station_001"]
```

### 6. docker-compose.yml
```yaml
# docker-compose.yml
version: '3.8'
services:
  privacyflow:
    build: .
    devices:
      - /dev/video0:/dev/video0  # Camera access
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
    environment:
      - STATION_ID=station_001
    restart: unless-stopped
```

### 7. Systemd Service (for Raspberry Pi)
```ini
# /etc/systemd/system/privacyflow.service
[Unit]
Description=PrivacyFlow Person Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/privacyflow
ExecStart=/home/pi/privacyflow/venv/bin/python main.py --config config.yaml --station-id station_001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 🗄️ Database Notes

### Current Implementation: SQLite In-Memory ✅
Your current `EphemeralDatabase` uses SQLite in-memory, which is **correct for privacy**:

```python
# Already implemented in src/database.py
self._conn = sqlite3.connect(':memory:', check_same_thread=False)
```

**No additional DB setup needed** - the in-memory approach ensures:
- Data is lost on restart (privacy feature)
- Fast operations
- No external dependencies

### Optional: Persistent Aggregate Storage
If you want daily summaries to persist (they don't contain PII):

```python
# Option 1: File-based SQLite for summaries only
persist_db = EphemeralDatabase(persist_path="/data/aggregates.db")

# Option 2: Add to config.yaml
output:
  persist_aggregates: true
  aggregates_path: "/data/aggregates.db"
```

---

## 📊 Optional Enhancements

### 8. Web Dashboard
Create `src/dashboard/` with simple HTML/JS:

```
src/dashboard/
├── index.html          # Main dashboard
├── static/
│   ├── js/app.js       # Fetch data from API
│   └── css/style.css
```

### 9. Monitoring & Alerts
```python
# src/monitoring.py
class SystemMonitor:
    def check_purge_executed(self) -> bool:
        """Verify daily purge ran successfully."""
        pass
    
    def send_alert(self, message: str) -> None:
        """Send alert via email/webhook."""
        pass
```

### 10. Data Export CLI
```python
# scripts/export_data.py
"""Export aggregate data for analysis."""
def export_daily_summaries(start_date, end_date, output_file):
    pass
```

---

## 🔒 Security Checklist

- [ ] Ensure camera feed is never saved to disk
- [ ] Verify raw images are not logged
- [ ] Test daily purge actually deletes all data
- [ ] Verify tokens cannot be reversed
- [ ] Test salt rotation breaks cross-day tracking
- [ ] Audit log contains no PII

---

## 📋 Implementation Priority

### Phase 1: MVP (Must Have)
1. ⬜ Camera integration (`src/camera.py`)
2. ⬜ Main entry point (`main.py`)
3. ⬜ Real person detection (update `detector.py`)
4. ⬜ Download/integrate pre-trained models

### Phase 2: Deployment
5. ⬜ Dockerfile
6. ⬜ Systemd service file
7. ⬜ Installation script for Raspberry Pi

### Phase 3: Polish
8. ⬜ Web dashboard
9. ⬜ Monitoring & alerts
10. ⬜ Documentation updates

---

## 🛠️ Quick Start for Code Agent

```bash
# Clone and setup
git clone <repo>
cd privacyflow
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download models (creates placeholders - need real models)
python scripts/download_models.py --lightweight

# Run tests
pytest tests/ -v

# Run examples (works with placeholder data)
python examples/basic_usage.py

# TODO: Implement camera.py and main.py before running real system
```

---

## Summary

| Category | Status | Action Needed |
|----------|--------|---------------|
| Core Logic | ✅ Done | - |
| Database | ✅ Done (in-memory SQLite) | Optional: persistent aggregates |
| Privacy Mechanisms | ✅ Done | Verify in production |
| ML Models | ❌ Placeholder | Download/train real models |
| Camera Integration | ❌ Missing | Implement `camera.py` |
| Entry Point | ❌ Missing | Create `main.py` |
| Real Detection | ❌ Placeholder | Update `detector.py` |
| Deployment | ❌ Missing | Add Docker/systemd |
| Dashboard | ❌ Optional | Nice to have |
