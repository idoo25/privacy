"""
PrivacyFlow - Main System Controller

Privacy-preserving person detection and flow analysis system.
"""

from datetime import datetime
from typing import Optional, Dict, Any
import yaml
import logging

from .privacy_token import PrivacyTokenGenerator
from .detector import PersonDetector
from .trajectory import TrajectoryBuilder
from .database import EphemeralDatabase
from .scheduler import DailyPurgeScheduler, AuditLogger
from .analytics import Analytics
from .data_structures import DetectionEvent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('privacyflow')


class PrivacyFlow:
    """Main system controller."""
    
    def __init__(self, config_path: str):
        """
        Initialize system with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.token_generator = PrivacyTokenGenerator()
        self.detector = PersonDetector(self.config)
        self.trajectory_builder = TrajectoryBuilder(self.config)
        self.database = EphemeralDatabase()
        self.analytics = Analytics(self.database, self.trajectory_builder)
        
        # Initialize scheduler for daily purge
        self._init_scheduler()
        
        # State
        self._running = False
        self._audit_logger = AuditLogger()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
    
    def _init_scheduler(self) -> None:
        """Initialize the daily purge scheduler."""
        purge_time = self.config.get('privacy', {}).get('purge_time', '00:00')
        
        def purge_callback() -> int:
            """Callback for daily purge."""
            # Clear database
            deleted = self.database.purge_daily_data()
            
            # Rotate salt
            self.token_generator.rotate_salt()
            
            # Clear trajectory builder
            self.trajectory_builder.clear()
            
            return deleted
        
        self.scheduler = DailyPurgeScheduler(
            purge_callback=purge_callback,
            halt_callback=self.emergency_halt,
            purge_time=purge_time
        )
    
    def start(self) -> None:
        """Start detection pipeline."""
        logger.info("Starting PrivacyFlow system")
        
        # Start scheduler
        self.scheduler.start()
        
        self._running = True
        
        logger.info("PrivacyFlow system started")
    
    def stop(self) -> None:
        """Stop detection and cleanup."""
        logger.info("Stopping PrivacyFlow system")
        
        self._running = False
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Close database
        self.database.close()
        
        logger.info("PrivacyFlow system stopped")
    
    def emergency_halt(self) -> None:
        """Emergency halt for critical failures."""
        logger.critical("EMERGENCY HALT - Privacy protection failure")
        self._running = False
        self.scheduler.stop()
    
    def process_frame(self, frame: 'np.ndarray', 
                     station_id: str,
                     camera_id: str) -> int:
        """
        Process a single frame for person detection.
        
        Args:
            frame: Input image frame
            station_id: Station identifier
            camera_id: Camera identifier
        
        Returns:
            Number of persons detected
        """
        if not self._running:
            return 0
        
        # Detect persons
        detections = self.detector.detect(frame)
        
        for detection in detections:
            # Extract features
            features = self.detector.extract_features(frame, detection)
            
            # Generate anonymous token
            token = self.token_generator.generate_token(
                features.face_embedding,
                features.body_proportions,
                features.appearance_features
            )
            
            # Create detection event
            event = DetectionEvent(
                token=token,
                timestamp=datetime.now(),
                station_id=station_id,
                camera_id=camera_id,
                confidence=detection.confidence,
                direction=None  # Would be determined by tracking logic
            )
            
            # Add to trajectory builder
            self.trajectory_builder.add_detection(event)
            
            # Store in database
            self.database.insert_detection(event)
        
        return len(detections)
    
    def get_current_count(self, station_id: str) -> int:
        """
        Get current person count at station.
        
        Args:
            station_id: Station identifier
        
        Returns:
            Current person count
        """
        return self.trajectory_builder.count_unique_tokens(station_id=station_id)
    
    def get_hourly_stats(self, station_id: str, hour: int) -> dict:
        """
        Get aggregate statistics for specific hour.
        
        Args:
            station_id: Station identifier
            hour: Hour of day (0-23)
        
        Returns:
            Dictionary with hourly statistics
        """
        from datetime import date
        today = date.today()
        aggregates = self.database.get_hourly_aggregates(station_id, today)
        
        for agg in aggregates:
            if agg.hour_start.hour == hour:
                return {
                    "station_id": station_id,
                    "hour": hour,
                    "unique_visitors": agg.unique_tokens,
                    "entries": agg.entry_count,
                    "exits": agg.exit_count,
                    "avg_dwell_time": agg.avg_dwell_time
                }
        
        return {
            "station_id": station_id,
            "hour": hour,
            "unique_visitors": 0,
            "entries": 0,
            "exits": 0,
            "avg_dwell_time": None
        }
    
    def export_daily_summary(self, date_str: str) -> dict:
        """
        Export anonymized daily summary.
        
        Args:
            date_str: Date string (YYYY-MM-DD)
        
        Returns:
            Dictionary with daily summary (aggregate only)
        """
        from datetime import date
        target_date = date.fromisoformat(date_str)
        device_id = self.config.get('system', {}).get('device_id', 'unknown')
        
        summary = self.database.get_daily_summary(device_id, target_date)
        
        if summary:
            return {
                "date": date_str,
                "station_id": summary.station_id,
                "total_unique_visitors": summary.total_unique_visitors,
                "peak_hour": summary.peak_hour,
                "avg_visit_duration": summary.avg_visit_duration,
                "round_trip_percentage": summary.round_trip_percentage
            }
        
        return {
            "date": date_str,
            "station_id": device_id,
            "total_unique_visitors": 0,
            "peak_hour": None,
            "avg_visit_duration": None,
            "round_trip_percentage": None
        }
