"""
Analytics Module

Aggregate statistics and reporting with privacy guarantees.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Callable
import json

from .data_structures import HourlyAggregate, DailySummary
from .database import EphemeralDatabase
from .trajectory import TrajectoryBuilder


class Analytics:
    """Analytics engine for aggregate statistics."""
    
    def __init__(self, database: EphemeralDatabase, 
                 trajectory_builder: TrajectoryBuilder):
        """
        Initialize analytics engine.
        
        Args:
            database: Database instance
            trajectory_builder: TrajectoryBuilder instance
        """
        self.db = database
        self.trajectory_builder = trajectory_builder
    
    @staticmethod
    def find_peak_hours(station_id: str, 
                       date_range: tuple,
                       database: Optional[EphemeralDatabase] = None) -> Dict[str, Any]:
        """
        Find peak hours for a station.
        
        Args:
            station_id: Station identifier
            date_range: Tuple of (start_date, end_date) as strings
            database: Optional database instance
        
        Returns:
            Dictionary with peak hour information
        """
        # Placeholder implementation
        return {
            "station_id": station_id,
            "peak_hour": 8,  # 8 AM
            "avg_count": 156,
            "date_range": date_range
        }
    
    @staticmethod
    def get_popular_routes(date_str: str, 
                          min_travelers: int = 10,
                          trajectory_builder: Optional[TrajectoryBuilder] = None) -> List[Dict[str, Any]]:
        """
        Get popular routes (aggregate only).
        
        Args:
            date_str: Date string (YYYY-MM-DD)
            min_travelers: Minimum travelers for privacy (default 10)
            trajectory_builder: Optional trajectory builder instance
        
        Returns:
            List of popular routes
        """
        if trajectory_builder is None:
            return []
        
        routes = {}
        for traj in trajectory_builder.get_trajectories(min_length=2):
            stations = traj.stations_visited
            for i in range(len(stations) - 1):
                route = (stations[i], stations[i + 1])
                routes[route] = routes.get(route, 0) + 1
        
        # Filter by minimum travelers for privacy
        popular = [
            {"from": route[0], "to": route[1], "count": count}
            for route, count in routes.items()
            if count >= min_travelers
        ]
        
        return sorted(popular, key=lambda x: x['count'], reverse=True)
    
    def calculate_hourly_aggregates(self, station_id: str, 
                                    target_date: date) -> List[HourlyAggregate]:
        """
        Calculate hourly aggregates for a station.
        
        Args:
            station_id: Station identifier
            target_date: Date to calculate aggregates for
        
        Returns:
            List of HourlyAggregate objects
        """
        aggregates = []
        
        for hour in range(24):
            start_time = datetime.combine(target_date, datetime.min.time().replace(hour=hour))
            end_time = start_time + timedelta(hours=1)
            
            # Get events in this hour
            events = self.trajectory_builder.get_events_in_window(start_time, end_time)
            station_events = [e for e in events if e.station_id == station_id]
            
            if station_events:
                unique_tokens = len(set(e.token for e in station_events))
                entry_count = sum(1 for e in station_events if e.direction == 'entry')
                exit_count = sum(1 for e in station_events if e.direction == 'exit')
                
                aggregate = HourlyAggregate(
                    station_id=station_id,
                    hour_start=start_time,
                    unique_tokens=unique_tokens,
                    entry_count=entry_count,
                    exit_count=exit_count,
                    avg_dwell_time=None  # Would need more complex calculation
                )
                
                aggregates.append(aggregate)
                self.db.insert_hourly_aggregate(aggregate)
        
        return aggregates
    
    def calculate_daily_summary(self, station_id: str, 
                               target_date: date) -> DailySummary:
        """
        Calculate daily summary for a station.
        
        Args:
            station_id: Station identifier
            target_date: Date to summarize
        
        Returns:
            DailySummary object
        """
        # Get hourly aggregates
        hourly = self.db.get_hourly_aggregates(station_id, target_date)
        
        if not hourly:
            # Calculate them first
            hourly = self.calculate_hourly_aggregates(station_id, target_date)
        
        # Find peak hour
        peak_hour = 0
        peak_count = 0
        total_visitors = 0
        
        for agg in hourly:
            total_visitors += agg.unique_tokens
            if agg.unique_tokens > peak_count:
                peak_count = agg.unique_tokens
                peak_hour = agg.hour_start.hour
        
        # Get round trip percentage from trajectory builder
        round_trip_pct = self.trajectory_builder.get_round_trip_percentage()
        
        summary = DailySummary(
            date=target_date,
            station_id=station_id,
            total_unique_visitors=total_visitors,
            peak_hour=peak_hour,
            avg_visit_duration=0.0,  # Would need more complex calculation
            round_trip_percentage=round_trip_pct
        )
        
        # Store in database
        self.db.insert_daily_summary(summary)
        
        return summary


class OccupancyMonitor:
    """Real-time occupancy monitoring for capacity management."""
    
    def __init__(self, station_id: str, 
                 max_capacity: int,
                 alert_threshold: float = 0.8):
        """
        Initialize occupancy monitor.
        
        Args:
            station_id: Station identifier
            max_capacity: Maximum capacity
            alert_threshold: Threshold for alerts (0.0-1.0)
        """
        self.station_id = station_id
        self.max_capacity = max_capacity
        self.alert_threshold = alert_threshold
        self._current_count = 0
        self._threshold_callback: Optional[Callable] = None
        self._running = False
    
    @property
    def current_count(self) -> int:
        """Get current person count."""
        return self._current_count
    
    @property
    def occupancy_ratio(self) -> float:
        """Get current occupancy ratio."""
        return self._current_count / self.max_capacity
    
    def on_threshold_exceeded(self, callback: Callable[[int, int], None]) -> Callable:
        """
        Decorator for threshold exceeded callback.
        
        Args:
            callback: Function to call when threshold exceeded
        
        Returns:
            The callback function
        """
        self._threshold_callback = callback
        return callback
    
    def update_count(self, count: int) -> None:
        """
        Update the current count.
        
        Args:
            count: New person count
        """
        self._current_count = count
        
        # Check threshold
        if self.occupancy_ratio >= self.alert_threshold:
            if self._threshold_callback:
                self._threshold_callback(self._current_count, self.max_capacity)
    
    def increment(self) -> None:
        """Increment count (person entered)."""
        self.update_count(self._current_count + 1)
    
    def decrement(self) -> None:
        """Decrement count (person exited)."""
        if self._current_count > 0:
            self.update_count(self._current_count - 1)
    
    def start(self) -> None:
        """Start the monitor."""
        self._running = True
    
    def stop(self) -> None:
        """Stop the monitor."""
        self._running = False


class PlanningReport:
    """Generate planning reports with aggregate statistics only."""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize report with data.
        
        Args:
            data: Report data
        """
        self.data = data
    
    @staticmethod
    def generate(stations: List[str],
                date_range: tuple,
                metrics: List[str],
                database: Optional[EphemeralDatabase] = None,
                trajectory_builder: Optional[TrajectoryBuilder] = None) -> 'PlanningReport':
        """
        Generate planning report (aggregates only).
        
        Args:
            stations: List of station IDs
            date_range: Tuple of (start_date, end_date)
            metrics: List of metrics to include
            database: Optional database instance
            trajectory_builder: Optional trajectory builder
        
        Returns:
            PlanningReport instance
        """
        report_data = {
            "stations": stations,
            "date_range": date_range,
            "metrics": {},
            "generated_at": datetime.now().isoformat()
        }
        
        # Placeholder metrics
        for metric in metrics:
            if metric == "daily_unique_visitors":
                report_data["metrics"][metric] = {
                    station: 0 for station in stations
                }
            elif metric == "hourly_distribution":
                report_data["metrics"][metric] = {
                    station: [0] * 24 for station in stations
                }
            elif metric == "route_popularity":
                report_data["metrics"][metric] = []
            elif metric == "round_trip_percentage":
                report_data["metrics"][metric] = 0.0
            elif metric == "average_journey_time":
                report_data["metrics"][metric] = 0.0
        
        return PlanningReport(report_data)
    
    def export(self, filename: str) -> None:
        """
        Export report to file.
        
        Args:
            filename: Output filename
        """
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(self.data, f, indent=2)
        elif filename.endswith('.pdf'):
            # Placeholder for PDF export
            # In production, would use reportlab or similar
            with open(filename.replace('.pdf', '.json'), 'w') as f:
                json.dump(self.data, f, indent=2)
        else:
            with open(filename, 'w') as f:
                json.dump(self.data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return report as dictionary."""
        return self.data
