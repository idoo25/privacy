"""
PrivacyFlow - Privacy-Preserving Person Detection System

A low-cost, privacy-first person detection and flow analysis system
designed for public transportation analytics.

Privacy by Design, not Privacy by Promise.
"""

from privacyflow.core import PrivacyFlow
from privacyflow.models.detection import DetectionEvent, DailyTrajectory
from privacyflow.models.token import PrivacyTokenGenerator
from privacyflow.tracking.detector import PersonDetector
from privacyflow.tracking.trajectory import TrajectoryBuilder
from privacyflow.camera import CameraCapture, CameraConfig, list_cameras, get_camera_info
from privacyflow.detectors import YOLOv8Detector, MoveNetPoseEstimator

__version__ = "0.1.0"
__all__ = [
    "PrivacyFlow",
    "DetectionEvent",
    "DailyTrajectory",
    "PrivacyTokenGenerator",
    "PersonDetector",
    "TrajectoryBuilder",
    "CameraCapture",
    "CameraConfig",
    "list_cameras",
    "get_camera_info",
    "YOLOv8Detector",
    "MoveNetPoseEstimator",
]
