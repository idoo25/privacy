"""
Database Storage - Ephemeral by Design

All tables are in-memory SQLite.
Automatically cleared at 00:00 via scheduled task.
"""

import sqlite3
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from privacyflow.models.detection import DetectionEvent


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages ephemeral SQLite database for detection storage.
    
    Key privacy features:
    - In-memory database (no persistent files)
    - Automatic daily purge at 00:00
    - Only aggregate statistics persist beyond midnight
    """
    
    # SQL Schema
    SCHEMA = """
    -- All tables are in-memory SQLite
    -- Automatically cleared at 00:00 via scheduled task

    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token TEXT NOT NULL,              -- 16-char hashed token
        timestamp DATETIME NOT NULL,
        station_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        confidence REAL NOT NULL,
        direction TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_token ON detections(token);
    CREATE INDEX IF NOT EXISTS idx_station_time ON detections(station_id, timestamp);

    CREATE TABLE IF NOT EXISTS hourly_aggregates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT NOT NULL,
        hour_start DATETIME NOT NULL,
        unique_tokens INTEGER NOT NULL,   -- Count of unique people
        entry_count INTEGER NOT NULL,
        exit_count INTEGER NOT NULL,
        avg_dwell_time REAL,
        
        UNIQUE(station_id, hour_start)
    );

    -- This table stores ONLY aggregate statistics
    -- No individual tracking data is retained after aggregation
    CREATE TABLE IF NOT EXISTS daily_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL,
        station_id TEXT NOT NULL,
        total_unique_visitors INTEGER,
        peak_hour INTEGER,
        avg_visit_duration REAL,
        round_trip_percentage REAL,
        
        -- This is the ONLY data that persists beyond midnight
        -- Contains no personally identifiable information
        UNIQUE(date, station_id)
    );
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize database manager.
        
        Args:
            db_path: Database path. Use ":memory:" for in-memory (recommended).
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize database with schema."""
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(self.SCHEMA)
        self._connection.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        try:
            yield self._connection
        except Exception as e:
            self._connection.rollback()
            raise e
    
    def insert_detection(self, event: DetectionEvent) -> int:
        """
        Insert a detection event.
        
        Args:
            event: DetectionEvent to insert
        
        Returns:
            ID of inserted row
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO detections (token, timestamp, station_id, camera_id, confidence, direction)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.token,
                    event.timestamp.isoformat(),
                    event.station_id,
                    event.camera_id,
                    event.confidence,
                    event.direction
                )
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_detections(self, 
                       station_id: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[DetectionEvent]:
        """
        Get detection events with optional filters.
        
        Args:
            station_id: Filter by station
            start_time: Filter by start time
            end_time: Filter by end time
        
        Returns:
            List of DetectionEvent objects
        """
        query = "SELECT * FROM detections WHERE 1=1"
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
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        return [
            DetectionEvent(
                token=row['token'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                station_id=row['station_id'],
                camera_id=row['camera_id'],
                confidence=row['confidence'],
                direction=row['direction']
            )
            for row in rows
        ]
    
    def get_unique_token_count(self, station_id: str, hour: Optional[int] = None) -> int:
        """
        Get count of unique tokens at a station.
        
        Args:
            station_id: Station identifier
            hour: Optional hour filter (0-23)
        
        Returns:
            Count of unique tokens
        """
        query = "SELECT COUNT(DISTINCT token) FROM detections WHERE station_id = ?"
        params = [station_id]
        
        if hour is not None:
            query += " AND strftime('%H', timestamp) = ?"
            params.append(f"{hour:02d}")
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()[0]
    
    def insert_hourly_aggregate(self, 
                                 station_id: str,
                                 hour_start: datetime,
                                 unique_tokens: int,
                                 entry_count: int,
                                 exit_count: int,
                                 avg_dwell_time: Optional[float] = None) -> int:
        """Insert hourly aggregate statistics."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO hourly_aggregates 
                (station_id, hour_start, unique_tokens, entry_count, exit_count, avg_dwell_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (station_id, hour_start.isoformat(), unique_tokens, entry_count, exit_count, avg_dwell_time)
            )
            conn.commit()
            return cursor.lastrowid
    
    def insert_daily_summary(self,
                             summary_date: date,
                             station_id: str,
                             total_unique_visitors: int,
                             peak_hour: int,
                             avg_visit_duration: float,
                             round_trip_percentage: float) -> int:
        """
        Insert daily summary (this data persists beyond midnight).
        
        This is aggregate-only data with no PII.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO daily_summary 
                (date, station_id, total_unique_visitors, peak_hour, avg_visit_duration, round_trip_percentage)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (summary_date.isoformat(), station_id, total_unique_visitors, peak_hour, avg_visit_duration, round_trip_percentage)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_daily_summary(self, summary_date: date, station_id: Optional[str] = None) -> List[Dict]:
        """Get daily summary data."""
        query = "SELECT * FROM daily_summary WHERE date = ?"
        params = [summary_date.isoformat()]
        
        if station_id:
            query += " AND station_id = ?"
            params.append(station_id)
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def purge_detections(self) -> int:
        """
        Delete all raw detection data.
        
        This is the PRIMARY privacy mechanism.
        Called at midnight during daily purge.
        
        Returns:
            Number of records deleted
        """
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM detections")
            count = cursor.fetchone()[0]
            
            conn.execute("DELETE FROM detections")
            conn.execute("DELETE FROM hourly_aggregates")
            conn.commit()
        
        logger.info(f"Purged {count} detection records")
        return count
    
    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


class DailyPurge:
    """
    Handles daily data purge - the core privacy guarantee.
    
    This task MUST run at midnight. If it fails, the system
    MUST halt to prevent privacy violations.
    """
    
    def __init__(self, 
                 db: DatabaseManager,
                 token_generator=None,
                 trajectory_builder=None,
                 on_failure=None):
        """
        Initialize daily purge handler.
        
        Args:
            db: DatabaseManager instance
            token_generator: PrivacyTokenGenerator to rotate salt
            trajectory_builder: TrajectoryBuilder to clear
            on_failure: Callback for purge failure (should halt system)
        """
        self.db = db
        self.token_generator = token_generator
        self.trajectory_builder = trajectory_builder
        self.on_failure = on_failure
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute daily purge.
        
        This is the PRIMARY privacy mechanism.
        If this fails, the system MUST halt.
        
        Returns:
            Purge statistics
        """
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'records_deleted': 0,
                'new_salt_generated': False,
                'trajectory_cleared': False
            }
            
            # Delete all raw detections
            stats['records_deleted'] = self.db.purge_detections()
            
            # Reset token generator salt
            if self.token_generator is not None:
                self.token_generator.rotate_salt()
                stats['new_salt_generated'] = True
            
            # Clear in-memory caches
            if self.trajectory_builder is not None:
                self.trajectory_builder.clear()
                stats['trajectory_cleared'] = True
            
            logger.info("Daily purge completed", extra=stats)
            return stats
            
        except Exception as e:
            # CRITICAL: If purge fails, halt system
            logger.critical(f"PURGE FAILED - HALTING SYSTEM: {e}")
            
            if self.on_failure is not None:
                self.on_failure(e)
            else:
                raise RuntimeError(f"Daily purge failed: {e}. System must halt.") from e
