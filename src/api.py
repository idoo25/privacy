"""
REST API Module

All endpoints return ONLY aggregate data.
No individual tracking information is exposed.
"""

from datetime import datetime, date
from typing import Optional

try:
    from flask import Flask, jsonify, request
except ImportError:
    Flask = None

from .data_structures import DailySummary
from .database import EphemeralDatabase
from .trajectory import TrajectoryBuilder
from .analytics import Analytics


def create_app(database: EphemeralDatabase,
               trajectory_builder: TrajectoryBuilder) -> 'Flask':
    """
    Create Flask application with API endpoints.
    
    Args:
        database: Database instance
        trajectory_builder: TrajectoryBuilder instance
    
    Returns:
        Flask application
    """
    if Flask is None:
        raise ImportError("Flask is required for API. Install with: pip install flask")
    
    app = Flask(__name__)
    analytics = Analytics(database, trajectory_builder)
    
    @app.get("/api/v1/stations/<station_id>/current")
    def get_current_count(station_id: str):
        """
        Get current person count at station.
        
        Returns:
            {
                "station_id": "station_001",
                "timestamp": "2024-01-15T14:30:00Z",
                "current_count": 42,
                "trend": "increasing"  # or "decreasing", "stable"
            }
        """
        # Get count from trajectory builder
        current_count = trajectory_builder.count_unique_tokens(station_id=station_id)
        
        # Determine trend (would need historical data in production)
        trend = "stable"
        
        return jsonify({
            "station_id": station_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "current_count": current_count,
            "trend": trend
        })
    
    @app.get("/api/v1/stations/<station_id>/hourly")
    def get_hourly_stats(station_id: str):
        """
        Get hourly aggregate statistics.
        
        Returns:
            {
                "station_id": "station_001",
                "date": "2024-01-15",
                "hours": [
                    {
                        "hour": 8,
                        "unique_visitors": 156,
                        "entries": 98,
                        "exits": 58,
                        "avg_dwell_minutes": 12.5
                    },
                    ...
                ]
            }
        """
        date_str = request.args.get('date', date.today().isoformat())
        target_date = date.fromisoformat(date_str)
        
        # Get hourly aggregates
        hourly = database.get_hourly_aggregates(station_id, target_date)
        
        hours_data = []
        for agg in hourly:
            hours_data.append({
                "hour": agg.hour_start.hour,
                "unique_visitors": agg.unique_tokens,
                "entries": agg.entry_count,
                "exits": agg.exit_count,
                "avg_dwell_minutes": agg.avg_dwell_time or 0.0
            })
        
        return jsonify({
            "station_id": station_id,
            "date": date_str,
            "hours": hours_data
        })
    
    @app.get("/api/v1/flow/matrix")
    def get_flow_matrix():
        """
        Get station-to-station flow matrix (aggregate only).
        
        Returns:
            {
                "date": "2024-01-15",
                "stations": ["A", "B", "C"],
                "matrix": [
                    [0, 150, 75],   # From A to A, B, C
                    [120, 0, 90],   # From B to A, B, C
                    [80, 95, 0]     # From C to A, B, C
                ]
            }
        """
        date_str = request.args.get('date', date.today().isoformat())
        
        # Get flow matrix from trajectory builder
        flow_matrix = trajectory_builder.get_flow_matrix()
        
        # Get list of stations
        stations = sorted(set(
            event.station_id 
            for event in trajectory_builder._events
        ))
        
        return jsonify({
            "date": date_str,
            "stations": stations,
            "matrix": flow_matrix.tolist()
        })
    
    @app.get("/api/v1/health")
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    return app


class APIServer:
    """API server wrapper."""
    
    def __init__(self, database: EphemeralDatabase,
                 trajectory_builder: TrajectoryBuilder,
                 host: str = "0.0.0.0",
                 port: int = 5000):
        """
        Initialize API server.
        
        Args:
            database: Database instance
            trajectory_builder: TrajectoryBuilder instance
            host: Host to bind to
            port: Port to listen on
        """
        self.database = database
        self.trajectory_builder = trajectory_builder
        self.host = host
        self.port = port
        self.app = None
    
    def start(self) -> None:
        """Start the API server."""
        self.app = create_app(self.database, self.trajectory_builder)
        self.app.run(host=self.host, port=self.port)
    
    def stop(self) -> None:
        """Stop the API server."""
        # Flask doesn't have a built-in way to stop gracefully
        pass
