"""
Database Module

Ephemeral SQLite storage with automatic daily purge.
"""

import sqlite3
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from .data_structures import DetectionEvent, HourlyAggregate, DailySummary


class EphemeralDatabase:
    """
    In-memory SQLite database for detection storage.
    
    All data is automatically cleared at 00:00 via scheduled task.
    Only aggregate statistics (daily_summary) persist beyond midnight.
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize ephemeral database.
        
        Args:
            persist_path: Optional path for persistent aggregate storage.
                         If None, uses in-memory database.
        """
        self.persist_path = persist_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        if self.persist_path:
            self._conn = sqlite3.connect(self.persist_path, check_same_thread=False)
        else:
            self._conn = sqlite3.connect(':memory:', check_same_thread=False)
        
        self._conn.row_factory = sqlite3.Row
        
        # Create tables
        self._conn.executescript("""
            -- Detections table (cleared at midnight)
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                station_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                direction TEXT
            );
            
            -- Indexes for fast lookups
            CREATE INDEX IF NOT EXISTS idx_token ON detections(token);
            CREATE INDEX IF NOT EXISTS idx_station_time ON detections(station_id, timestamp);
            
            -- Hourly aggregates (cleared at midnight)
            CREATE TABLE IF NOT EXISTS hourly_aggregates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                hour_start DATETIME NOT NULL,
                unique_tokens INTEGER NOT NULL,
                entry_count INTEGER NOT NULL,
                exit_count INTEGER NOT NULL,
                avg_dwell_time REAL,
                UNIQUE(station_id, hour_start)
            );
            
            -- Daily summary (persists beyond midnight)
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                station_id TEXT NOT NULL,
                total_unique_visitors INTEGER,
                peak_hour INTEGER,
                avg_visit_duration REAL,
                round_trip_percentage REAL,
                UNIQUE(date, station_id)
            );
        """)
        self._conn.commit()
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
    
    def insert_detection(self, event: DetectionEvent) -> int:
        """
        Insert a detection event.
        
        Args:
            event: DetectionEvent to insert
        
        Returns:
            Row ID of inserted event
        """
        cursor = self._conn.execute("""
            INSERT INTO detections (token, timestamp, station_id, camera_id, confidence, direction)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event.token,
            event.timestamp.isoformat(),
            event.station_id,
            event.camera_id,
            event.confidence,
            event.direction
        ))
        self._conn.commit()
        return cursor.lastrowid
    
    def get_detections_by_token(self, token: str) -> List[DetectionEvent]:
        """Get all detections for a token."""
        cursor = self._conn.execute("""
            SELECT * FROM detections WHERE token = ? ORDER BY timestamp
        """, (token,))
        
        return [self._row_to_event(row) for row in cursor.fetchall()]
    
    def get_detections_by_station(self, station_id: str, 
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> List[DetectionEvent]:
        """Get detections for a station within time range."""
        query = "SELECT * FROM detections WHERE station_id = ?"
        params = [station_id]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp"
        
        cursor = self._conn.execute(query, params)
        return [self._row_to_event(row) for row in cursor.fetchall()]
    
    def count_unique_tokens(self, station_id: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> int:
        """Count unique tokens in time range."""
        query = "SELECT COUNT(DISTINCT token) FROM detections WHERE 1=1"
        params = []
        
        if station_id:
            query += " AND station_id = ?"
            params.append(station_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        cursor = self._conn.execute(query, params)
        return cursor.fetchone()[0]
    
    def insert_hourly_aggregate(self, aggregate: HourlyAggregate) -> None:
        """Insert or update hourly aggregate."""
        self._conn.execute("""
            INSERT OR REPLACE INTO hourly_aggregates 
            (station_id, hour_start, unique_tokens, entry_count, exit_count, avg_dwell_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            aggregate.station_id,
            aggregate.hour_start.isoformat(),
            aggregate.unique_tokens,
            aggregate.entry_count,
            aggregate.exit_count,
            aggregate.avg_dwell_time
        ))
        self._conn.commit()
    
    def get_hourly_aggregates(self, station_id: str, 
                              date_filter: Optional[date] = None) -> List[HourlyAggregate]:
        """Get hourly aggregates for a station."""
        query = "SELECT * FROM hourly_aggregates WHERE station_id = ?"
        params = [station_id]
        
        if date_filter:
            query += " AND DATE(hour_start) = ?"
            params.append(date_filter.isoformat())
        
        query += " ORDER BY hour_start"
        
        cursor = self._conn.execute(query, params)
        return [self._row_to_hourly_aggregate(row) for row in cursor.fetchall()]
    
    def insert_daily_summary(self, summary: DailySummary) -> None:
        """Insert or update daily summary."""
        self._conn.execute("""
            INSERT OR REPLACE INTO daily_summary 
            (date, station_id, total_unique_visitors, peak_hour, avg_visit_duration, round_trip_percentage)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            summary.date.isoformat() if isinstance(summary.date, date) else summary.date,
            summary.station_id,
            summary.total_unique_visitors,
            summary.peak_hour,
            summary.avg_visit_duration,
            summary.round_trip_percentage
        ))
        self._conn.commit()
    
    def get_daily_summary(self, station_id: str, 
                          date_filter: date) -> Optional[DailySummary]:
        """Get daily summary for a station and date."""
        cursor = self._conn.execute("""
            SELECT * FROM daily_summary WHERE station_id = ? AND date = ?
        """, (station_id, date_filter.isoformat()))
        
        row = cursor.fetchone()
        if row:
            return self._row_to_daily_summary(row)
        return None
    
    def purge_daily_data(self) -> int:
        """
        Delete all detection data.
        
        This is the PRIMARY privacy mechanism.
        
        Returns:
            Number of records deleted
        """
        # Count records to be deleted
        cursor = self._conn.execute("SELECT COUNT(*) FROM detections")
        count = cursor.fetchone()[0]
        
        # Delete all raw detections
        self._conn.execute("DELETE FROM detections")
        
        # Also clear hourly aggregates (they're intermediate data)
        self._conn.execute("DELETE FROM hourly_aggregates")
        
        self._conn.commit()
        return count
    
    def _row_to_event(self, row: sqlite3.Row) -> DetectionEvent:
        """Convert database row to DetectionEvent."""
        return DetectionEvent(
            token=row['token'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            station_id=row['station_id'],
            camera_id=row['camera_id'],
            confidence=row['confidence'],
            direction=row['direction']
        )
    
    def _row_to_hourly_aggregate(self, row: sqlite3.Row) -> HourlyAggregate:
        """Convert database row to HourlyAggregate."""
        return HourlyAggregate(
            station_id=row['station_id'],
            hour_start=datetime.fromisoformat(row['hour_start']),
            unique_tokens=row['unique_tokens'],
            entry_count=row['entry_count'],
            exit_count=row['exit_count'],
            avg_dwell_time=row['avg_dwell_time']
        )
    
    def _row_to_daily_summary(self, row: sqlite3.Row) -> DailySummary:
        """Convert database row to DailySummary."""
        return DailySummary(
            date=datetime.fromisoformat(row['date']) if isinstance(row['date'], str) else row['date'],
            station_id=row['station_id'],
            total_unique_visitors=row['total_unique_visitors'],
            peak_hour=row['peak_hour'],
            avg_visit_duration=row['avg_visit_duration'],
            round_trip_percentage=row['round_trip_percentage']
        )
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
