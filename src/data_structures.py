"""
Data Structures Module

Contains data classes for detection events and trajectories.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class DetectionEvent:
    """Single detection event - stored only until midnight."""
    
    token: str                    # 16-char anonymous token
    timestamp: datetime           # Detection time
    station_id: str              # Location identifier
    camera_id: str               # Camera identifier
    confidence: float            # Detection confidence
    direction: Optional[str] = None  # Entry/exit/passing
    
    # NOTE: Raw features are NEVER stored
    # Only the hashed token is persisted


@dataclass
class DailyTrajectory:
    """
    Reconstructed trajectory for a single token.
    Automatically deleted at 00:00.
    """
    
    token: str
    events: List[DetectionEvent] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    stations_visited: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.events:
            self._update_from_events()
    
    def _update_from_events(self):
        """Update derived fields from events."""
        if self.events:
            sorted_events = sorted(self.events, key=lambda e: e.timestamp)
            self.first_seen = sorted_events[0].timestamp
            self.last_seen = sorted_events[-1].timestamp
            self.stations_visited = [e.station_id for e in sorted_events]
    
    def add_event(self, event: DetectionEvent):
        """Add an event to the trajectory."""
        self.events.append(event)
        self._update_from_events()
    
    @property
    def is_round_trip(self) -> bool:
        """Check if trajectory starts and ends at same station."""
        if len(self.stations_visited) < 2:
            return False
        return self.stations_visited[0] == self.stations_visited[-1]


@dataclass
class Detection:
    """Detection result from person detector."""
    
    bbox: tuple  # (x, y, w, h)
    confidence: float
    face_crop: Optional['np.ndarray'] = None
    pose_keypoints: Optional['np.ndarray'] = None
    segmentation_mask: Optional['np.ndarray'] = None


@dataclass
class FeatureVector:
    """Combined feature vector for a detected person."""
    
    face_embedding: 'np.ndarray'  # 64-dim
    body_proportions: 'np.ndarray'  # 8-dim
    appearance_features: 'np.ndarray'  # 32-dim
    
    def to_tuple(self):
        """Return features as a tuple for token generation."""
        return (self.face_embedding, self.body_proportions, self.appearance_features)


@dataclass
class HourlyAggregate:
    """Hourly aggregate statistics."""
    
    station_id: str
    hour_start: datetime
    unique_tokens: int
    entry_count: int
    exit_count: int
    avg_dwell_time: Optional[float] = None


@dataclass
class DailySummary:
    """
    Daily summary statistics.
    This is the ONLY data that persists beyond midnight.
    Contains no personally identifiable information.
    """
    
    date: datetime
    station_id: str
    total_unique_visitors: int
    peak_hour: int
    avg_visit_duration: float
    round_trip_percentage: float
