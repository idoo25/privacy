"""
Trajectory Builder

Builds trajectories from detection events using token matching.
All data is ephemeral and deleted at midnight.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np

from privacyflow.models.detection import DetectionEvent, DailyTrajectory


class TrajectoryBuilder:
    """
    Builds trajectories from detection events.
    
    All trajectory data is ephemeral and automatically
    deleted during the daily purge at 00:00.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 temporal_window_seconds: int = 300):
        """
        Initialize trajectory builder.
        
        Args:
            similarity_threshold: Threshold for token matching (0-1)
            temporal_window_seconds: Max time gap for same trajectory (seconds)
        """
        self.similarity_threshold = similarity_threshold
        self.temporal_window = timedelta(seconds=temporal_window_seconds)
        
        # Storage (cleared at midnight)
        self._events: List[DetectionEvent] = []
        self._trajectories: Dict[str, DailyTrajectory] = {}
    
    def add_detection(self, event: DetectionEvent) -> None:
        """
        Add new detection event.
        
        Args:
            event: DetectionEvent to add
        """
        self._events.append(event)
        
        # Update or create trajectory
        if event.token in self._trajectories:
            self._trajectories[event.token].add_event(event)
        else:
            self._trajectories[event.token] = DailyTrajectory(
                token=event.token,
                events=[event]
            )
    
    def get_trajectories(self, min_length: int = 2) -> List[DailyTrajectory]:
        """
        Get all trajectories with minimum length.
        
        Args:
            min_length: Minimum number of events in trajectory
        
        Returns:
            List of DailyTrajectory objects
        """
        return [
            traj for traj in self._trajectories.values()
            if len(traj.events) >= min_length
        ]
    
    def get_trajectory(self, token: str) -> Optional[DailyTrajectory]:
        """
        Get trajectory for a specific token.
        
        Args:
            token: Person token
        
        Returns:
            DailyTrajectory or None if not found
        """
        return self._trajectories.get(token)
    
    def get_flow_matrix(self, stations: Optional[List[str]] = None) -> np.ndarray:
        """
        Get station-to-station flow matrix (aggregate only).
        
        Args:
            stations: List of station IDs (auto-discovered if None)
        
        Returns:
            Flow matrix where matrix[i][j] = count from station i to station j
        """
        if stations is None:
            # Auto-discover stations
            stations = sorted(set(
                event.station_id for event in self._events
            ))
        
        n_stations = len(stations)
        station_to_idx = {s: i for i, s in enumerate(stations)}
        
        flow_matrix = np.zeros((n_stations, n_stations), dtype=int)
        
        for trajectory in self._trajectories.values():
            if len(trajectory.stations_visited) >= 2:
                for i in range(len(trajectory.stations_visited) - 1):
                    from_station = trajectory.stations_visited[i]
                    to_station = trajectory.stations_visited[i + 1]
                    
                    if from_station in station_to_idx and to_station in station_to_idx:
                        from_idx = station_to_idx[from_station]
                        to_idx = station_to_idx[to_station]
                        flow_matrix[from_idx, to_idx] += 1
        
        return flow_matrix
    
    def get_station_counts(self) -> Dict[str, int]:
        """
        Get unique person counts per station.
        
        Returns:
            Dictionary mapping station_id to unique token count
        """
        station_tokens: Dict[str, set] = defaultdict(set)
        
        for event in self._events:
            station_tokens[event.station_id].add(event.token)
        
        return {station: len(tokens) for station, tokens in station_tokens.items()}
    
    def get_hourly_stats(self, station_id: str, hour: int) -> Dict:
        """
        Get aggregate statistics for a specific hour.
        
        Args:
            station_id: Station identifier
            hour: Hour of day (0-23)
        
        Returns:
            Dictionary with aggregate statistics
        """
        hour_events = [
            e for e in self._events
            if e.station_id == station_id and e.timestamp.hour == hour
        ]
        
        unique_tokens = set(e.token for e in hour_events)
        entries = sum(1 for e in hour_events if e.direction == 'entry')
        exits = sum(1 for e in hour_events if e.direction == 'exit')
        
        return {
            'station_id': station_id,
            'hour': hour,
            'unique_visitors': len(unique_tokens),
            'entries': entries,
            'exits': exits,
            'total_detections': len(hour_events)
        }
    
    def get_current_count(self, station_id: str, window_minutes: int = 5) -> int:
        """
        Get current person count at station.
        
        Args:
            station_id: Station identifier
            window_minutes: Time window to consider
        
        Returns:
            Number of unique persons in the window
        """
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)
        
        recent_tokens = set(
            e.token for e in self._events
            if e.station_id == station_id and e.timestamp >= window_start
        )
        
        return len(recent_tokens)
    
    def clear(self) -> int:
        """
        Clear all trajectory data (used during daily purge).
        
        Returns:
            Number of events deleted
        """
        count = len(self._events)
        self._events = []
        self._trajectories = {}
        return count
    
    @property
    def event_count(self) -> int:
        """Get total number of stored events."""
        return len(self._events)
    
    @property
    def trajectory_count(self) -> int:
        """Get total number of trajectories."""
        return len(self._trajectories)
