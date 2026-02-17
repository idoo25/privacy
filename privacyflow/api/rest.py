"""
REST API Endpoints for PrivacyFlow

All endpoints return ONLY aggregate data.
No individual tracking information is exposed.
"""

from datetime import datetime, date
from typing import Dict, Any, Optional

from privacyflow.core import PrivacyFlow


class PrivacyFlowAPI:
    """
    REST API handler for PrivacyFlow.
    
    All endpoints return ONLY aggregate data.
    No individual tracking information is exposed.
    """
    
    def __init__(self, privacyflow: PrivacyFlow):
        """
        Initialize API with PrivacyFlow instance.
        
        Args:
            privacyflow: PrivacyFlow system instance
        """
        self.pf = privacyflow
    
    def get_current_count(self, station_id: str) -> Dict[str, Any]:
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
        current = self.pf.get_current_count(station_id)
        
        # Calculate trend (simple comparison with previous count)
        # In a real implementation, this would track historical counts
        trend = "stable"
        
        return {
            "station_id": station_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "current_count": current,
            "trend": trend
        }
    
    def get_hourly_stats(self, station_id: str, query_date: Optional[str] = None) -> Dict[str, Any]:
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
        if query_date is None:
            query_date = date.today().isoformat()
        
        hours = []
        for hour in range(24):
            stats = self.pf.get_hourly_stats(station_id, hour)
            hours.append({
                "hour": hour,
                "unique_visitors": stats['unique_visitors'],
                "entries": stats['entries'],
                "exits": stats['exits'],
                "avg_dwell_minutes": None  # Would be calculated from trajectory data
            })
        
        return {
            "station_id": station_id,
            "date": query_date,
            "hours": hours
        }
    
    def get_flow_matrix(self, query_date: Optional[str] = None) -> Dict[str, Any]:
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
        if query_date is None:
            query_date = date.today().isoformat()
        
        summary = self.pf.export_daily_summary()
        
        if summary.get('flow_matrix'):
            return {
                "date": query_date,
                "stations": summary['flow_matrix']['stations'],
                "matrix": summary['flow_matrix']['matrix']
            }
        else:
            return {
                "date": query_date,
                "stations": [],
                "matrix": []
            }
    
    def get_daily_summary(self, query_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily summary for all stations.
        
        Returns aggregate-only data with no PII.
        """
        if query_date is None:
            query_date = date.today().isoformat()
        
        return self.pf.export_daily_summary()


def create_app(privacyflow: PrivacyFlow):
    """
    Create a simple Flask-like app for the API.
    
    This is a minimal implementation. In production,
    use Flask, FastAPI, or similar framework.
    
    Args:
        privacyflow: PrivacyFlow system instance
    
    Returns:
        API application object
    """
    api = PrivacyFlowAPI(privacyflow)
    
    # Route definitions (for documentation purposes)
    routes = {
        '/api/v1/stations/{station_id}/current': api.get_current_count,
        '/api/v1/stations/{station_id}/hourly': api.get_hourly_stats,
        '/api/v1/flow/matrix': api.get_flow_matrix,
        '/api/v1/daily-summary': api.get_daily_summary,
    }
    
    return {
        'api': api,
        'routes': routes
    }
