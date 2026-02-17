"""
Data Structures Module

Contains data classes for detection events and trajectories.
Optimized for efficient operations with large datasets.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import bisect


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
    
    def __lt__(self, other: 'DetectionEvent') -> bool:
        """Enable sorting by timestamp."""
        return self.timestamp < other.timestamp


@dataclass
class DailyTrajectory:
    """
    Reconstructed trajectory for a single token.
    Automatically deleted at 00:00.
    
    Optimized with:
    - Bisect-based sorted insertion (O(log n) vs O(n log n))
    - Cached derived values
    - Lazy updates for computed properties
    """
    
    token: str
    events: List[DetectionEvent] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    stations_visited: List[str] = field(default_factory=list)
    _is_sorted: bool = field(default=True, repr=False)
    _round_trip_cached: Optional[bool] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.events:
            self._ensure_sorted()
            self._update_bounds()
    
    def _ensure_sorted(self):
        """Ensure events are sorted by timestamp."""
        if not self._is_sorted:
            self.events.sort(key=lambda e: e.timestamp)
            self._is_sorted = True
            self.stations_visited = [e.station_id for e in self.events]
    
    def _update_bounds(self):
        """Update first_seen and last_seen from events."""
        if self.events:
            self._ensure_sorted()
            self.first_seen = self.events[0].timestamp
            self.last_seen = self.events[-1].timestamp
            self.stations_visited = [e.station_id for e in self.events]
    
    def add_event(self, event: DetectionEvent):
        """
        Add an event to the trajectory using bisect for O(log n) insertion.
        Utilizes the __lt__ operator on DetectionEvent for direct comparison.
        """
        # Use bisect with the event's __lt__ method - O(log n) search
        insert_pos = bisect.bisect_left(self.events, event)
        self.events.insert(insert_pos, event)
        
        # Update stations_visited at the correct position
        self.stations_visited.insert(insert_pos, event.station_id)
        
        # Update bounds efficiently
        if self.first_seen is None or event.timestamp < self.first_seen:
            self.first_seen = event.timestamp
        if self.last_seen is None or event.timestamp > self.last_seen:
            self.last_seen = event.timestamp
        
        # Invalidate cached round trip status
        self._round_trip_cached = None
    
    @property
    def is_round_trip(self) -> bool:
        """Check if trajectory starts and ends at same station (cached)."""
        if self._round_trip_cached is not None:
            return self._round_trip_cached
        
        if len(self.stations_visited) < 2:
            self._round_trip_cached = False
        else:
            self._round_trip_cached = self.stations_visited[0] == self.stations_visited[-1]
        return self._round_trip_cached


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
