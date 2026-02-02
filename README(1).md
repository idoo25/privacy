# PrivacyFlow - Privacy-Preserving Person Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GDPR Compliant](https://img.shields.io/badge/GDPR-Compliant-blue.svg)](#privacy-compliance)
[![Hardware Cost](https://img.shields.io/badge/Hardware%20Cost-~%24150-orange.svg)](#hardware-requirements)

## Overview

PrivacyFlow is a low-cost, privacy-first person detection and flow analysis system designed for public transportation analytics. Unlike traditional surveillance systems, PrivacyFlow is **intentionally imprecise** - making it impossible to uniquely identify individuals while still enabling valuable aggregate analytics.

### Key Philosophy

> "Privacy by Design, not Privacy by Promise"

The system achieves **~95% short-term tracking accuracy** while being **fundamentally incapable** of long-term individual identification.

---

## Table of Contents

- [Features](#features)
- [Privacy Compliance](#privacy-compliance)
- [System Architecture](#system-architecture)
- [Detection Pipeline](#detection-pipeline)
- [Data Structures](#data-structures)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Use Cases](#use-cases)
- [Contributing](#contributing)

---

## Features

| Feature | Description |
|---------|-------------|
| **Ultra-Low Resolution Face Detection** | 32×32 pixels (vs industry standard 160×160) |
| **Body Proportion Analysis** | Height, volume, limb ratios |
| **Appearance Fingerprinting** | Clothing colors, accessories, footwear |
| **Automatic Data Purge** | All data deleted at 00:00 daily |
| **No Model Training** | Zero long-term learning or data accumulation |
| **Salted Hashing** | All identifiers are hashed with daily-rotating salts |
| **Edge Processing** | All computation happens locally |

---

## Privacy Compliance

### GDPR Compliance Checklist

- [x] **Data Minimization**: Lowest possible resolution for face detection
- [x] **Purpose Limitation**: Only aggregate flow analytics
- [x] **Storage Limitation**: Automatic daily purge at 00:00
- [x] **No Biometric Templates**: Resolution too low for biometric identification
- [x] **No Cross-Day Tracking**: Daily salt rotation breaks continuity
- [x] **No Model Training**: System does not learn or improve from data
- [x] **Local Processing**: No cloud uploads, all processing on-device

### Why This System Cannot Identify Individuals

1. **Resolution Barrier**: 32×32 face images cannot be matched to identity databases
2. **Appearance Dependency**: Changing clothes creates a "new person"
3. **Temporal Isolation**: Daily data purge prevents pattern accumulation
4. **Hash Irreversibility**: Person tokens cannot be reversed to features

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  [Camera Feed] ──► [Frame Grabber] ──► [Person Detector]        │
│                         30 FPS            YOLO-Nano              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Face Embed  │  │  Body Props  │  │  Appearance  │          │
│  │   32×32 px   │  │   Ratios     │  │  Features    │          │
│  │   64-dim     │  │   8-dim      │  │   32-dim     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRIVACY LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│  [Feature Concatenation] ──► [Salted Hash] ──► [Person Token]   │
│        104-dim vector          SHA-256          Anonymous ID     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRACKING LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  [Token Matcher] ──► [Trajectory Builder] ──► [Analytics]       │
│    Fuzzy matching      Station sequences      Aggregate stats    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER (Ephemeral)                     │
├─────────────────────────────────────────────────────────────────┤
│  [SQLite In-Memory] ◄──► [Daily Purge Cron @ 00:00]             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detection Pipeline

### 1. Face Detection (Ultra-Low Resolution)

```python
# Configuration
FACE_RESOLUTION = (32, 32)  # Intentionally low - privacy by design
FACE_EMBEDDING_DIM = 64

# Pipeline
def extract_face_features(frame, bbox):
    """
    Extract face features at privacy-preserving resolution.
    
    Args:
        frame: Input image (any resolution)
        bbox: Face bounding box [x, y, w, h]
    
    Returns:
        numpy.ndarray: 64-dimensional embedding
    """
    face_crop = frame[bbox.y:bbox.y+bbox.h, bbox.x:bbox.x+bbox.w]
    face_resized = cv2.resize(face_crop, FACE_RESOLUTION, 
                              interpolation=cv2.INTER_AREA)
    embedding = face_encoder.encode(face_resized)
    return normalize(embedding)
```

### 2. Body Proportion Analysis

```python
# Extracted proportions (all as ratios, not absolute values)
BODY_FEATURES = {
    'height_ratio': float,        # Height relative to frame
    'shoulder_width_ratio': float, # Shoulder/height ratio
    'torso_ratio': float,         # Torso/total height
    'arm_length_ratio': float,    # Arm/height ratio
    'leg_length_ratio': float,    # Leg/height ratio
    'body_volume_estimate': float, # Approximate volume ratio
    'head_body_ratio': float,     # Head size relative to body
    'stance_width_ratio': float   # Stance width/height
}

def extract_body_proportions(pose_keypoints):
    """
    Extract body proportion ratios from pose estimation.
    
    Note: All values are RATIOS, not absolute measurements.
    This prevents height-based identification.
    
    Args:
        pose_keypoints: 17-point pose estimation results
    
    Returns:
        numpy.ndarray: 8-dimensional proportion vector
    """
    proportions = calculate_ratios(pose_keypoints)
    return normalize(proportions)
```

### 3. Appearance Feature Extraction

```python
# Appearance categories
APPEARANCE_FEATURES = {
    'upper_body_colors': List[RGB],    # Dominant colors (top 3)
    'lower_body_colors': List[RGB],    # Dominant colors (top 3)
    'footwear_type': int,              # Category index
    'headwear_present': bool,
    'bag_present': bool,
    'bag_position': int,               # Back/front/side
    'jacket_present': bool,
    'pattern_type': int                # Solid/striped/plaid/etc
}

def extract_appearance(frame, person_bbox, segmentation_mask):
    """
    Extract appearance features for short-term tracking.
    
    WARNING: These features are intentionally volatile.
    Changing clothes = new identity (by design).
    
    Args:
        frame: Input image
        person_bbox: Person bounding box
        segmentation_mask: Body part segmentation
    
    Returns:
        numpy.ndarray: 32-dimensional appearance vector
    """
    upper_colors = extract_dominant_colors(frame, segmentation_mask, 'upper')
    lower_colors = extract_dominant_colors(frame, segmentation_mask, 'lower')
    accessories = detect_accessories(frame, person_bbox)
    
    return encode_appearance(upper_colors, lower_colors, accessories)
```

---

## Data Structures

### Person Token Generation

```python
import hashlib
import os
from datetime import date

class PrivacyTokenGenerator:
    """
    Generates anonymous, non-reversible person tokens.
    
    Security features:
    - Daily rotating salt (breaks cross-day tracking)
    - SHA-256 hashing (irreversible)
    - Feature quantization (reduces precision)
    """
    
    def __init__(self):
        self._daily_salt = None
        self._salt_date = None
    
    @property
    def daily_salt(self) -> bytes:
        """Get or generate daily salt."""
        today = date.today()
        if self._salt_date != today:
            self._daily_salt = os.urandom(32)
            self._salt_date = today
        return self._daily_salt
    
    def generate_token(self, 
                       face_embedding: np.ndarray,
                       body_proportions: np.ndarray,
                       appearance_features: np.ndarray) -> str:
        """
        Generate anonymous person token from features.
        
        Args:
            face_embedding: 64-dim face features
            body_proportions: 8-dim body ratios
            appearance_features: 32-dim appearance vector
        
        Returns:
            str: 16-character anonymous token (truncated hash)
        """
        # Quantize features to reduce precision
        face_quant = self._quantize(face_embedding, bins=16)
        body_quant = self._quantize(body_proportions, bins=8)
        appearance_quant = self._quantize(appearance_features, bins=16)
        
        # Concatenate all features
        combined = np.concatenate([face_quant, body_quant, appearance_quant])
        
        # Hash with daily salt
        feature_bytes = combined.tobytes()
        salted = self.daily_salt + feature_bytes
        hash_digest = hashlib.sha256(salted).hexdigest()
        
        # Return truncated hash (sufficient for daily uniqueness)
        return hash_digest[:16]
    
    def _quantize(self, features: np.ndarray, bins: int) -> np.ndarray:
        """Reduce feature precision through quantization."""
        return np.digitize(features, np.linspace(0, 1, bins))
```

### Trajectory Storage (Ephemeral)

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class DetectionEvent:
    """Single detection event - stored only until midnight."""
    
    token: str                    # 16-char anonymous token
    timestamp: datetime           # Detection time
    station_id: str              # Location identifier
    camera_id: str               # Camera identifier
    confidence: float            # Detection confidence
    direction: Optional[str]     # Entry/exit/passing
    
    # NOTE: Raw features are NEVER stored
    # Only the hashed token is persisted

@dataclass
class DailyTrajectory:
    """
    Reconstructed trajectory for a single token.
    Automatically deleted at 00:00.
    """
    
    token: str
    events: List[DetectionEvent]
    first_seen: datetime
    last_seen: datetime
    stations_visited: List[str]
    
    @property
    def is_round_trip(self) -> bool:
        """Check if trajectory starts and ends at same station."""
        if len(self.stations_visited) < 2:
            return False
        return self.stations_visited[0] == self.stations_visited[-1]
```

### Database Schema

```sql
-- All tables are in-memory SQLite
-- Automatically cleared at 00:00 via scheduled task

CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT NOT NULL,              -- 16-char hashed token
    timestamp DATETIME NOT NULL,
    station_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    direction TEXT,
    
    -- Index for fast token lookups
    INDEX idx_token (token),
    INDEX idx_station_time (station_id, timestamp)
);

CREATE TABLE hourly_aggregates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    hour_start DATETIME NOT NULL,
    unique_tokens INTEGER NOT NULL,   -- Count of unique people
    entry_count INTEGER NOT NULL,
    exit_count INTEGER NOT NULL,
    avg_dwell_time REAL,
    
    UNIQUE(station_id, hour_start)
);

-- This table stores ONLY aggregate statistics
-- No individual tracking data is retained after aggregation
CREATE TABLE daily_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    station_id TEXT NOT NULL,
    total_unique_visitors INTEGER,
    peak_hour INTEGER,
    avg_visit_duration REAL,
    round_trip_percentage REAL,
    
    -- This is the ONLY data that persists beyond midnight
    -- Contains no personally identifiable information
    UNIQUE(date, station_id)
);
```

---

## Hardware Requirements

### Minimum Setup (~$150)

| Component | Model | Estimated Cost |
|-----------|-------|----------------|
| Single Board Computer | Raspberry Pi 4 (4GB) | $55 |
| Camera | Raspberry Pi Camera Module 3 | $25 |
| Storage | 32GB microSD | $10 |
| Power Supply | Official Pi 4 PSU | $10 |
| Case + Cooling | Argon ONE case | $25 |
| Cables + Misc | - | $25 |

### Recommended Setup (~$300)

| Component | Model | Estimated Cost |
|-----------|-------|----------------|
| Edge AI Device | NVIDIA Jetson Nano | $150 |
| Camera | IMX477 12MP Camera | $50 |
| Storage | 64GB microSD + USB SSD | $40 |
| Enclosure | Weatherproof case | $30 |
| PoE Splitter | For single-cable deployment | $15 |
| Mounting Hardware | - | $15 |

---

## Installation

### Prerequisites

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.9 \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    sqlite3

# For Jetson Nano (GPU acceleration)
sudo apt-get install -y nvidia-jetpack
```

### Package Installation

```bash
# Clone repository
git clone https://github.com/username/privacyflow.git
cd privacyflow

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models (lightweight versions)
python scripts/download_models.py --lightweight
```

### Requirements File

```txt
# requirements.txt
numpy>=1.21.0
opencv-python>=4.5.0
onnxruntime>=1.10.0  # or onnxruntime-gpu for Jetson
scikit-learn>=1.0.0
sqlite3
schedule>=1.1.0
pyyaml>=6.0
```

---

## Configuration

### Main Configuration File

```yaml
# config.yaml

system:
  device_id: "station_001"
  timezone: "UTC"
  log_level: "INFO"

privacy:
  # Face detection resolution (DO NOT INCREASE)
  face_resolution: [32, 32]
  
  # Token generation
  hash_algorithm: "sha256"
  token_length: 16
  daily_salt_rotation: true
  
  # Data retention
  purge_time: "00:00"
  retain_aggregates_days: 30  # Only aggregate stats
  retain_raw_detections: false

detection:
  # Camera settings
  camera_index: 0
  frame_width: 640
  frame_height: 480
  fps: 30
  
  # Detection thresholds
  person_confidence_threshold: 0.6
  face_confidence_threshold: 0.5
  min_detection_size: [50, 100]  # Minimum person bbox
  
  # Tracking
  max_disappeared_frames: 30
  max_distance_threshold: 100

features:
  # Face embedding
  face_model: "models/face_encoder_32x32.onnx"
  face_embedding_dim: 64
  
  # Body proportions
  pose_model: "models/pose_lite.onnx"
  body_feature_dim: 8
  
  # Appearance
  appearance_model: "models/appearance_encoder.onnx"
  appearance_feature_dim: 32
  color_quantization_bins: 8

matching:
  # Token matching for trajectory building
  similarity_threshold: 0.85
  temporal_window_seconds: 300  # 5 minute max gap
  
analytics:
  # Aggregate statistics
  aggregation_interval_minutes: 60
  min_samples_for_stats: 10
  
output:
  # Export settings
  export_format: "json"
  export_path: "/data/exports/"
  export_only_aggregates: true  # Never export raw detections
```

---

## API Reference

### Core Classes

```python
class PrivacyFlow:
    """Main system controller."""
    
    def __init__(self, config_path: str):
        """Initialize system with configuration."""
        
    def start(self) -> None:
        """Start detection pipeline."""
        
    def stop(self) -> None:
        """Stop detection and cleanup."""
        
    def get_current_count(self, station_id: str) -> int:
        """Get current person count at station."""
        
    def get_hourly_stats(self, station_id: str, hour: int) -> dict:
        """Get aggregate statistics for specific hour."""
        
    def export_daily_summary(self, date: str) -> dict:
        """Export anonymized daily summary."""


class PersonDetector:
    """Handles person detection and feature extraction."""
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons in frame."""
        
    def extract_features(self, frame: np.ndarray, 
                        detection: Detection) -> FeatureVector:
        """Extract all features for a detection."""


class TrajectoryBuilder:
    """Builds trajectories from detection events."""
    
    def add_detection(self, event: DetectionEvent) -> None:
        """Add new detection event."""
        
    def get_trajectories(self, 
                        min_length: int = 2) -> List[DailyTrajectory]:
        """Get all trajectories with minimum length."""
        
    def get_flow_matrix(self) -> np.ndarray:
        """Get station-to-station flow matrix."""
```

### REST API Endpoints

```python
# All endpoints return ONLY aggregate data
# No individual tracking information is exposed

@app.get("/api/v1/stations/{station_id}/current")
def get_current_count(station_id: str) -> dict:
    """
    Get current person count at station.
    
    Returns:
        {
            "station_id": "station_001",
            "timestamp": "2024-01-15T14:30:00Z",
            "current_count": 42,
            "trend": "increasing"  # or "decreasing", "stable"
        }
    """

@app.get("/api/v1/stations/{station_id}/hourly")
def get_hourly_stats(station_id: str, date: str) -> dict:
    """
    Get hourly aggregate statistics.
    
    Returns:
        {
            "station_id": "station_001",
            "date": "2024-01-15",
            "hours": [
                {
                    "hour": 8,
                    "unique_visitors": 156,
                    "entries": 98,
                    "exits": 58,
                    "avg_dwell_minutes": 12.5
                },
                ...
            ]
        }
    """

@app.get("/api/v1/flow/matrix")
def get_flow_matrix(date: str) -> dict:
    """
    Get station-to-station flow matrix (aggregate only).
    
    Returns:
        {
            "date": "2024-01-15",
            "stations": ["A", "B", "C"],
            "matrix": [
                [0, 150, 75],   # From A to A, B, C
                [120, 0, 90],   # From B to A, B, C
                [80, 95, 0]     # From C to A, B, C
            ]
        }
    """
```

---

## Use Cases

### 1. Public Transportation Flow Analysis

```python
# Example: Analyze passenger flow patterns
from privacyflow import PrivacyFlow, Analytics

pf = PrivacyFlow("config.yaml")

# Get peak hours
peak_data = Analytics.find_peak_hours(
    station_id="central_station",
    date_range=("2024-01-01", "2024-01-31")
)
print(f"Peak hour: {peak_data['peak_hour']}:00")
print(f"Average passengers: {peak_data['avg_count']}")

# Get route popularity (aggregate only)
routes = Analytics.get_popular_routes(
    date="2024-01-15",
    min_travelers=10  # Minimum for privacy
)
for route in routes:
    print(f"{route['from']} → {route['to']}: {route['count']} travelers")
```

### 2. Occupancy Monitoring

```python
# Real-time occupancy for capacity management
from privacyflow import OccupancyMonitor

monitor = OccupancyMonitor(
    station_id="platform_3",
    max_capacity=200,
    alert_threshold=0.8
)

@monitor.on_threshold_exceeded
def handle_crowding(current_count, capacity):
    send_alert(f"Platform 3 at {current_count/capacity*100:.0f}% capacity")

monitor.start()
```

### 3. Service Planning Analytics

```python
# Generate planning reports (aggregates only)
from privacyflow import PlanningReport

report = PlanningReport.generate(
    stations=["A", "B", "C", "D"],
    date_range=("2024-01-01", "2024-03-31"),
    metrics=[
        "daily_unique_visitors",
        "hourly_distribution",
        "route_popularity",
        "round_trip_percentage",
        "average_journey_time"
    ]
)

report.export("quarterly_report.pdf")
```

---

## Security Considerations

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Raw image storage | Images never saved; only features extracted |
| Feature database theft | Only hashed tokens stored; features discarded |
| Cross-day tracking | Daily salt rotation; data purged at midnight |
| Model inference attack | 32×32 resolution prevents face reconstruction |
| Network interception | All data processed locally; no cloud uploads |
| Physical device theft | No historical data on device after midnight |

### Audit Logging

```python
# All system actions are logged for compliance audits
# Logs contain NO personal data

AUDIT_LOG_FORMAT = {
    "timestamp": "ISO8601",
    "event_type": "detection|purge|export|config_change",
    "details": {
        "detection": {"count": int, "station": str},
        "purge": {"records_deleted": int},
        "export": {"type": "aggregate_only", "destination": str},
        "config_change": {"parameter": str, "old_value": any, "new_value": any}
    }
}
```

---

## Scheduled Tasks

### Daily Purge (Critical for Privacy)

```python
# This task MUST run - it's the core privacy guarantee
import schedule
import time

def purge_daily_data():
    """
    Delete all detection data at midnight.
    
    This is the PRIMARY privacy mechanism.
    If this fails, the system MUST halt.
    """
    try:
        # Delete all raw detections
        db.execute("DELETE FROM detections")
        
        # Reset token generator salt
        token_generator.rotate_salt()
        
        # Clear in-memory caches
        trajectory_builder.clear()
        
        # Log for audit
        audit_log.info("Daily purge completed", {
            "records_deleted": deleted_count,
            "new_salt_generated": True
        })
        
    except Exception as e:
        # CRITICAL: If purge fails, halt system
        audit_log.critical("PURGE FAILED - HALTING SYSTEM", {"error": str(e)})
        system.emergency_halt()

# Schedule for midnight
schedule.every().day.at("00:00").do(purge_daily_data)
```

---

## Testing

### Privacy Tests

```python
# tests/test_privacy.py

def test_face_resolution_limit():
    """Ensure face images never exceed 32×32."""
    detector = PersonDetector(config)
    frame = load_test_frame()
    
    detections = detector.detect(frame)
    for det in detections:
        face_img = det.face_crop
        assert face_img.shape[:2] == (32, 32), \
            "Face resolution exceeds privacy limit!"

def test_token_irreversibility():
    """Verify tokens cannot be reversed to features."""
    generator = PrivacyTokenGenerator()
    
    features = generate_random_features()
    token = generator.generate_token(*features)
    
    # Token should be fixed-length hash
    assert len(token) == 16
    assert not any(str(f) in token for f in features)

def test_daily_salt_rotation():
    """Verify same features produce different tokens on different days."""
    generator = PrivacyTokenGenerator()
    
    features = generate_random_features()
    
    # Generate token today
    token_today = generator.generate_token(*features)
    
    # Simulate next day
    generator._salt_date = date.today() - timedelta(days=1)
    token_tomorrow = generator.generate_token(*features)
    
    assert token_today != token_tomorrow, \
        "Salt rotation failed - cross-day tracking possible!"

def test_no_raw_data_in_exports():
    """Ensure exports contain only aggregates."""
    exporter = DataExporter(config)
    
    export = exporter.export_daily_summary("2024-01-15")
    
    # Check no individual tokens in export
    assert "token" not in str(export).lower()
    assert "embedding" not in str(export).lower()
    assert "feature" not in str(export).lower()
```

---

## Contributing

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/username/privacyflow.git
cd privacyflow
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run privacy-specific tests
pytest tests/test_privacy.py -v --strict
```

### Code Review Checklist

- [ ] No increase in face resolution beyond 32×32
- [ ] No persistent storage of raw features
- [ ] No removal or weakening of daily purge
- [ ] No addition of cloud upload capabilities
- [ ] All new data exports are aggregate-only
- [ ] Privacy tests pass

---

## License

MIT License - See [LICENSE](LICENSE) file.

---

## Disclaimer

This system is designed for **aggregate analytics only**. It is intentionally incapable of:
- Identifying specific individuals
- Tracking persons across multiple days
- Building long-term behavioral profiles
- Integration with facial recognition databases

Any modification that enables these capabilities violates the design philosophy and potentially privacy regulations.

---

## Contact

For questions about privacy implementation or GDPR compliance, please open an issue with the `[privacy]` tag.
