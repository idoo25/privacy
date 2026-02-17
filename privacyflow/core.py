"""
PrivacyFlow Core - Main System Controller

Privacy by Design, not Privacy by Promise.
"""

import logging
import yaml
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, List

import schedule

from privacyflow.models.token import PrivacyTokenGenerator
from privacyflow.models.detection import DetectionEvent
from privacyflow.tracking.detector import PersonDetector
from privacyflow.tracking.trajectory import TrajectoryBuilder
from privacyflow.database.storage import DatabaseManager, DailyPurge


logger = logging.getLogger(__name__)


class PrivacyFlow:
    """
    Main system controller for PrivacyFlow.
    
    A low-cost, privacy-first person detection and flow analysis system
    designed for public transportation analytics.
    
    Key Features:
    - Ultra-low resolution face detection (32x32 pixels)
    - Daily data purge at 00:00
    - No long-term learning or data accumulation
    - All processing happens locally
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize system with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (alternative to file)
        """
        self.config = self._load_config(config_path, config)
        self._running = False
        self._halted = False
        
        # Initialize components
        self.token_generator = PrivacyTokenGenerator()
        self.detector = PersonDetector(self.config.get('features', {}))
        self.trajectory_builder = TrajectoryBuilder(
            similarity_threshold=self.config.get('matching', {}).get('similarity_threshold', 0.85),
            temporal_window_seconds=self.config.get('matching', {}).get('temporal_window_seconds', 300)
        )
        self.db = DatabaseManager(self.config.get('database', {}).get('path', ':memory:'))
        
        # Initialize daily purge
        self.daily_purge = DailyPurge(
            db=self.db,
            token_generator=self.token_generator,
            trajectory_builder=self.trajectory_builder,
            on_failure=self.emergency_halt
        )
        
        # Schedule daily purge
        purge_time = self.config.get('privacy', {}).get('purge_time', '00:00')
        schedule.every().day.at(purge_time).do(self.daily_purge.execute)
        
        logger.info(f"PrivacyFlow initialized with device_id: {self.config.get('system', {}).get('device_id', 'unknown')}")
    
    def _load_config(self, config_path: Optional[str], config: Optional[Dict]) -> Dict[str, Any]:
        """Load configuration from file or dictionary."""
        if config is not None:
            return config
        
        if config_path is not None:
            path = Path(config_path)
            if path.exists():
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
        
        # Return default configuration
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'system': {
                'device_id': 'default_device',
                'timezone': 'UTC',
                'log_level': 'INFO'
            },
            'privacy': {
                'face_resolution': [32, 32],
                'hash_algorithm': 'sha256',
                'token_length': 16,
                'daily_salt_rotation': True,
                'purge_time': '00:00',
                'retain_aggregates_days': 30,
                'retain_raw_detections': False
            },
            'detection': {
                'camera_index': 0,
                'frame_width': 640,
                'frame_height': 480,
                'fps': 30,
                'person_confidence_threshold': 0.6,
                'face_confidence_threshold': 0.5,
                'min_detection_size': [50, 100],
                'max_disappeared_frames': 30,
                'max_distance_threshold': 100
            },
            'features': {
                'face_embedding_dim': 64,
                'body_feature_dim': 8,
                'appearance_feature_dim': 32,
                'color_quantization_bins': 8
            },
            'matching': {
                'similarity_threshold': 0.85,
                'temporal_window_seconds': 300
            },
            'analytics': {
                'aggregation_interval_minutes': 60,
                'min_samples_for_stats': 10
            },
            'output': {
                'export_format': 'json',
                'export_only_aggregates': True
            }
        }
    
    def start(self) -> None:
        """Start detection pipeline."""
        if self._halted:
            raise RuntimeError("System has been halted due to privacy failure. Manual intervention required.")
        
        self._running = True
        logger.info("PrivacyFlow detection pipeline started")
    
    def stop(self) -> None:
        """Stop detection and cleanup."""
        self._running = False
        logger.info("PrivacyFlow detection pipeline stopped")
    
    def emergency_halt(self, error: Optional[Exception] = None) -> None:
        """
        Emergency halt - called when privacy guarantees cannot be maintained.
        
        This is a critical safety mechanism. If the daily purge fails,
        the system MUST stop to prevent privacy violations.
        """
        self._halted = True
        self._running = False
        logger.critical(f"EMERGENCY HALT: {error or 'Privacy guarantee failure'}")
    
    def process_frame(self, frame, station_id: str, camera_id: str) -> List[DetectionEvent]:
        """
        Process a single frame and return detection events.
        
        Args:
            frame: Input image (numpy array)
            station_id: Location identifier
            camera_id: Camera identifier
        
        Returns:
            List of DetectionEvent objects
        """
        if not self._running or self._halted:
            return []
        
        events = []
        
        # Detect persons
        detections = self.detector.detect(frame)
        
        for detection in detections:
            # Extract features
            features = self.detector.extract_features(frame, detection)
            
            # Generate privacy token
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
                confidence=detection.confidence
            )
            
            # Store event
            self.trajectory_builder.add_detection(event)
            self.db.insert_detection(event)
            
            events.append(event)
        
        # Run scheduled tasks (including daily purge)
        schedule.run_pending()
        
        return events
    
    def get_current_count(self, station_id: str) -> int:
        """
        Get current person count at station.
        
        Args:
            station_id: Location identifier
        
        Returns:
            Current count of unique persons
        """
        return self.trajectory_builder.get_current_count(station_id)
    
    def get_hourly_stats(self, station_id: str, hour: int) -> Dict:
        """
        Get aggregate statistics for specific hour.
        
        Args:
            station_id: Location identifier
            hour: Hour of day (0-23)
        
        Returns:
            Dictionary with aggregate statistics
        """
        return self.trajectory_builder.get_hourly_stats(station_id, hour)
    
    def export_daily_summary(self, export_date: Optional[date] = None) -> Dict:
        """
        Export anonymized daily summary.
        
        Only aggregate statistics are exported.
        No individual tracking information is included.
        
        Args:
            export_date: Date to export (default: today)
        
        Returns:
            Dictionary with anonymized summary
        """
        if export_date is None:
            export_date = date.today()
        
        # Get station counts
        station_counts = self.trajectory_builder.get_station_counts()
        
        # Get trajectories for round-trip calculation
        trajectories = self.trajectory_builder.get_trajectories(min_length=2)
        round_trips = sum(1 for t in trajectories if t.is_round_trip)
        round_trip_pct = (round_trips / len(trajectories) * 100) if trajectories else 0
        
        # Build summary
        summary = {
            'date': export_date.isoformat(),
            'stations': {},
            'flow_matrix': None
        }
        
        for station_id, count in station_counts.items():
            # Calculate peak hour
            hourly_counts = []
            for hour in range(24):
                stats = self.get_hourly_stats(station_id, hour)
                hourly_counts.append((hour, stats['unique_visitors']))
            
            peak_hour = max(hourly_counts, key=lambda x: x[1])[0] if hourly_counts else 0
            
            summary['stations'][station_id] = {
                'total_unique_visitors': count,
                'peak_hour': peak_hour,
                'round_trip_percentage': round_trip_pct
            }
        
        # Get flow matrix
        stations = sorted(station_counts.keys())
        if stations:
            flow_matrix = self.trajectory_builder.get_flow_matrix(stations)
            summary['flow_matrix'] = {
                'stations': stations,
                'matrix': flow_matrix.tolist()
            }
        
        return summary
    
    @property
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._running
    
    @property
    def is_halted(self) -> bool:
        """Check if system has been halted."""
        return self._halted
