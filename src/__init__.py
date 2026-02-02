# PrivacyFlow - Privacy-Preserving Person Detection System
"""
Privacy-first person detection and flow analysis system.
"""

from .privacyflow import PrivacyFlow
from .privacy_token import PrivacyTokenGenerator
from .data_structures import DetectionEvent, DailyTrajectory
from .trajectory import TrajectoryBuilder
from .analytics import Analytics, OccupancyMonitor, PlanningReport

__version__ = "1.0.0"
__all__ = [
    "PrivacyFlow",
    "PrivacyTokenGenerator", 
    "DetectionEvent",
    "DailyTrajectory",
    "TrajectoryBuilder",
    "Analytics",
    "OccupancyMonitor",
    "PlanningReport",
]
