"""
Trajectory Builder Module

Builds trajectories from detection events with temporal constraints.

Optimized with:
- Indexed lookups for O(1) station and time-based queries
- Vectorized flow matrix computation using numpy
- Cached statistics for frequently accessed metrics
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import bisect
import numpy as np

from .data_structures import DetectionEvent, DailyTrajectory


class TrajectoryBuilder:
    """
    Builds trajectories from detection events.
    
    Optimization Features:
    - Station index: O(1) lookup of events by station
    - Time index: O(log n) range queries by timestamp
    - Token index: O(1) lookup of events by token
    - Cached round trip count
    - Vectorized flow matrix computation
    """
    
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
        
        # Optimized indexes for fast lookups
        self._station_index: Dict[str, List[int]] = defaultdict(list)  # station_id -> event indices
        self._time_sorted_indices: List[int] = []  # indices sorted by timestamp
        self._timestamps: List[datetime] = []  # timestamps for binary search
        self._token_event_indices: Dict[str, List[int]] = defaultdict(list)  # token -> event indices
        
        # Cached statistics
        self._round_trip_count_cached: Optional[int] = None
        self._stations_set: Set[str] = set()
    
    def add_detection(self, event: DetectionEvent) -> None:
        """
        Add new detection event with index updates.
        
        Args:
            event: DetectionEvent to add
        """
        event_idx = len(self._events)
        self._events.append(event)
        
        # Update station index - O(1)
        self._station_index[event.station_id].append(event_idx)
        self._stations_set.add(event.station_id)
        
        # Update time index using bisect for sorted insertion - O(log n)
        insert_pos = bisect.bisect_left(self._timestamps, event.timestamp)
        self._timestamps.insert(insert_pos, event.timestamp)
        self._time_sorted_indices.insert(insert_pos, event_idx)
        
        # Update token index - O(1)
        self._token_event_indices[event.token].append(event_idx)
        
        # Update or create trajectory for this token
        if event.token in self._trajectories:
            self._trajectories[event.token].add_event(event)
        else:
            self._trajectories[event.token] = DailyTrajectory(
                token=event.token,
                events=[event]
            )
        
        # Invalidate cached statistics
        self._round_trip_count_cached = None
    
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
        Get station-to-station flow matrix using vectorized computation.
        
        Args:
            stations: List of station IDs (optional, auto-detected if not provided)
        
        Returns:
            2D numpy array of flow counts
        """
        # Auto-detect stations if not provided - use cached set
        if stations is None:
            stations = sorted(self._stations_set)
        
        n_stations = len(stations)
        if n_stations == 0:
            return np.zeros((0, 0), dtype=int)
        
        station_idx = {s: i for i, s in enumerate(stations)}
        
        # Collect all transitions for vectorized counting
        from_indices = []
        to_indices = []
        
        for traj in self._trajectories.values():
            visited = traj.stations_visited
            for i in range(len(visited) - 1):
                from_station = visited[i]
                to_station = visited[i + 1]
                if from_station in station_idx and to_station in station_idx:
                    from_indices.append(station_idx[from_station])
                    to_indices.append(station_idx[to_station])
        
        # Vectorized flow matrix construction using numpy bincount
        if from_indices:
            # Convert to numpy arrays
            from_arr = np.array(from_indices, dtype=int)
            to_arr = np.array(to_indices, dtype=int)
            
            # Use flat indices for bincount
            flat_indices = from_arr * n_stations + to_arr
            counts = np.bincount(flat_indices, minlength=n_stations * n_stations)
            flow_matrix = counts.reshape(n_stations, n_stations)
        else:
            flow_matrix = np.zeros((n_stations, n_stations), dtype=int)
        
        return flow_matrix
    
    def get_trajectory_by_token(self, token: str) -> Optional[DailyTrajectory]:
        """
        Get trajectory for a specific token - O(1).
        
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
        Get events within a time window using binary search - O(log n + k).
        
        Args:
            start_time: Window start
            end_time: Window end
        
        Returns:
            List of DetectionEvent objects
        """
        # Binary search for start and end positions - O(log n)
        start_pos = bisect.bisect_left(self._timestamps, start_time)
        end_pos = bisect.bisect_right(self._timestamps, end_time)
        
        # Retrieve events in range - O(k) where k is result size
        return [
            self._events[self._time_sorted_indices[i]]
            for i in range(start_pos, end_pos)
        ]
    
    def get_station_events(self, station_id: str) -> List[DetectionEvent]:
        """
        Get all events for a specific station using index - O(k).
        
        Args:
            station_id: Station identifier
        
        Returns:
            List of DetectionEvent objects
        """
        # Use station index for O(1) lookup of indices
        indices = self._station_index.get(station_id, [])
        return [self._events[i] for i in indices]
    
    def count_unique_tokens(self, 
                           station_id: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> int:
        """
        Count unique tokens using optimized index lookups.
        
        Args:
            station_id: Optional station filter
            start_time: Optional start time filter
            end_time: Optional end time filter
        
        Returns:
            Count of unique tokens
        """
        # Fast path: no filters
        if station_id is None and start_time is None and end_time is None:
            return len(self._trajectories)
        
        tokens = set()
        
        # Use appropriate index based on filters
        if station_id is not None and start_time is None and end_time is None:
            # Station-only filter: use station index
            for idx in self._station_index.get(station_id, []):
                tokens.add(self._events[idx].token)
        elif station_id is None and (start_time is not None or end_time is not None):
            # Time-only filter: use time index
            start_pos = 0 if start_time is None else bisect.bisect_left(self._timestamps, start_time)
            end_pos = len(self._timestamps) if end_time is None else bisect.bisect_right(self._timestamps, end_time)
            
            for i in range(start_pos, end_pos):
                tokens.add(self._events[self._time_sorted_indices[i]].token)
        else:
            # Combined filters: get events in time window then filter by station
            start = start_time if start_time is not None else (
                self._timestamps[0] if self._timestamps else datetime.min
            )
            end = end_time if end_time is not None else (
                self._timestamps[-1] if self._timestamps else datetime.max
            )
            events = self.get_events_in_window(start, end)
            for event in events:
                if station_id is None or event.station_id == station_id:
                    tokens.add(event.token)
        
        return len(tokens)
    
    def clear(self) -> None:
        """Clear all events, trajectories, and indexes (for daily purge)."""
        self._events.clear()
        self._trajectories.clear()
        self._station_index.clear()
        self._time_sorted_indices.clear()
        self._timestamps.clear()
        self._token_event_indices.clear()
        self._round_trip_count_cached = None
        self._stations_set.clear()
    
    def get_round_trip_count(self) -> int:
        """Count trajectories that are round trips (cached)."""
        if self._round_trip_count_cached is None:
            self._round_trip_count_cached = sum(
                1 for traj in self._trajectories.values() 
                if traj.is_round_trip
            )
        return self._round_trip_count_cached
    
    def get_round_trip_percentage(self) -> float:
        """Calculate percentage of round trip trajectories."""
        total = len(self._trajectories)
        if total == 0:
            return 0.0
        return (self.get_round_trip_count() / total) * 100
    
    def get_station_count(self) -> int:
        """Get count of unique stations - O(1)."""
        return len(self._stations_set)
    
    def get_all_stations(self) -> List[str]:
        """Get sorted list of all stations - O(n log n) for sorting."""
        return sorted(self._stations_set)
