"""
Trajectory Builder Module

Builds trajectories from detection events with temporal constraints.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np

from .data_structures import DetectionEvent, DailyTrajectory


class TrajectoryBuilder:
    """Builds trajectories from detection events."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trajectory builder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Matching configuration
        matching_config = self.config.get('matching', {})
        self.similarity_threshold = matching_config.get('similarity_threshold', 0.85)
        self.temporal_window_seconds = matching_config.get('temporal_window_seconds', 300)
        
        # Storage for events and trajectories
        self._events: List[DetectionEvent] = []
        self._trajectories: Dict[str, DailyTrajectory] = {}
    
    def add_detection(self, event: DetectionEvent) -> None:
        """
        Add new detection event.
        
        Args:
            event: DetectionEvent to add
        """
        self._events.append(event)
        
        # Update or create trajectory for this token
        if event.token in self._trajectories:
            self._trajectories[event.token].add_event(event)
        else:
            self._trajectories[event.token] = DailyTrajectory(
                token=event.token,
                events=[event]
            )
    
    def get_trajectories(self, 
                        min_length: int = 2) -> List[DailyTrajectory]:
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
    
    def get_flow_matrix(self, stations: Optional[List[str]] = None) -> np.ndarray:
        """
        Get station-to-station flow matrix.
        
        Args:
            stations: List of station IDs (optional, auto-detected if not provided)
        
        Returns:
            2D numpy array of flow counts
        """
        # Auto-detect stations if not provided
        if stations is None:
            stations = sorted(set(
                event.station_id 
                for event in self._events
            ))
        
        n_stations = len(stations)
        station_idx = {s: i for i, s in enumerate(stations)}
        
        # Initialize flow matrix
        flow_matrix = np.zeros((n_stations, n_stations), dtype=int)
        
        # Count flows from trajectories
        for traj in self._trajectories.values():
            visited = traj.stations_visited
            for i in range(len(visited) - 1):
                from_station = visited[i]
                to_station = visited[i + 1]
                if from_station in station_idx and to_station in station_idx:
                    from_idx = station_idx[from_station]
                    to_idx = station_idx[to_station]
                    flow_matrix[from_idx, to_idx] += 1
        
        return flow_matrix
    
    def get_trajectory_by_token(self, token: str) -> Optional[DailyTrajectory]:
        """
        Get trajectory for a specific token.
        
        Args:
            token: Person token
        
        Returns:
            DailyTrajectory or None
        """
        return self._trajectories.get(token)
    
    def get_events_in_window(self, 
                            start_time: datetime,
                            end_time: datetime) -> List[DetectionEvent]:
        """
        Get events within a time window.
        
        Args:
            start_time: Window start
            end_time: Window end
        
        Returns:
            List of DetectionEvent objects
        """
        return [
            event for event in self._events
            if start_time <= event.timestamp <= end_time
        ]
    
    def get_station_events(self, station_id: str) -> List[DetectionEvent]:
        """
        Get all events for a specific station.
        
        Args:
            station_id: Station identifier
        
        Returns:
            List of DetectionEvent objects
        """
        return [
            event for event in self._events
            if event.station_id == station_id
        ]
    
    def count_unique_tokens(self, 
                           station_id: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> int:
        """
        Count unique tokens, optionally filtered by station and time.
        
        Args:
            station_id: Optional station filter
            start_time: Optional start time filter
            end_time: Optional end time filter
        
        Returns:
            Count of unique tokens
        """
        tokens = set()
        for event in self._events:
            if station_id and event.station_id != station_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            tokens.add(event.token)
        return len(tokens)
    
    def clear(self) -> None:
        """Clear all events and trajectories (for daily purge)."""
        self._events.clear()
        self._trajectories.clear()
    
    def get_round_trip_count(self) -> int:
        """Count trajectories that are round trips."""
        return sum(1 for traj in self._trajectories.values() if traj.is_round_trip)
    
    def get_round_trip_percentage(self) -> float:
        """Calculate percentage of round trip trajectories."""
        total = len(self._trajectories)
        if total == 0:
            return 0.0
        return (self.get_round_trip_count() / total) * 100
