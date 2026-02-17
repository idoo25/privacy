"""Detection data structures - stored only until midnight."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class DetectionEvent:
    """
    Single detection event - stored only until midnight.
    
    NOTE: Raw features are NEVER stored.
    Only the hashed token is persisted.
    """
    
    token: str                    # 16-char anonymous token
    timestamp: datetime           # Detection time
    station_id: str              # Location identifier
    camera_id: str               # Camera identifier
    confidence: float            # Detection confidence
    direction: Optional[str] = None  # Entry/exit/passing


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
        """Update first/last seen and stations from events."""
        if self.events:
            self.events.sort(key=lambda e: e.timestamp)
            if self.first_seen is None:
                self.first_seen = self.events[0].timestamp
            if self.last_seen is None:
                self.last_seen = self.events[-1].timestamp
            if not self.stations_visited:
                self.stations_visited = [e.station_id for e in self.events]
    
    @property
    def is_round_trip(self) -> bool:
        """Check if trajectory starts and ends at same station."""
        if len(self.stations_visited) < 2:
            return False
        return self.stations_visited[0] == self.stations_visited[-1]
    
    def add_event(self, event: DetectionEvent) -> None:
        """Add a detection event to the trajectory."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.timestamp)
        self.first_seen = self.events[0].timestamp
        self.last_seen = self.events[-1].timestamp
        self.stations_visited = [e.station_id for e in self.events]
