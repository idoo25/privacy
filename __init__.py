# PrivacyFlow - Privacy-Preserving Person Detection System
"""
Privacy-first person detection and flow analysis system.

Core Modules:
- PrivacyFlow: Main system controller
- PrivacyTokenGenerator: Anonymous token generation with daily salt
- PersonDetector: Person detection and feature extraction
- TrajectoryBuilder: Trajectory building with optimized indexes
- StationGraph: Graph-based route analysis (DFS/BFS)
- EphemeralDatabase: In-memory SQLite with daily purge
- CameraCapture: Thread-safe camera capture

Example:
    from privacyflow import PrivacyFlow
    
    pf = PrivacyFlow("config.yaml")
    pf.start()
"""

from .privacyflow import PrivacyFlow
from .privacy_token import PrivacyTokenGenerator
from .data_structures import DetectionEvent, DailyTrajectory, FeatureVector
from .trajectory import TrajectoryBuilder
from .analytics import Analytics, OccupancyMonitor, PlanningReport
from .station_graph import StationGraph
from .database import EphemeralDatabase
from .detector import PersonDetector
from .camera import CameraCapture, MultiCameraCapture, list_available_cameras

__version__ = "1.0.0"
__author__ = "PrivacyFlow"

__all__ = [
    # Main controller
    "PrivacyFlow",
    
    # Privacy components
    "PrivacyTokenGenerator",
    
    # Data structures
    "DetectionEvent",
    "DailyTrajectory",
    "FeatureVector",
    
    # Processing
    "PersonDetector",
    "TrajectoryBuilder",
    "StationGraph",
    
    # Storage
    "EphemeralDatabase",
    
    # Analytics
    "Analytics",
    "OccupancyMonitor",
    "PlanningReport",
    
    # Camera
    "CameraCapture",
    "MultiCameraCapture",
    "list_available_cameras",
]
