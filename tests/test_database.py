"""
Tests for database storage and daily purge.
"""

import pytest
from datetime import datetime, date, timedelta

from privacyflow.models.detection import DetectionEvent
from privacyflow.models.token import PrivacyTokenGenerator
from privacyflow.database.storage import DatabaseManager, DailyPurge
from privacyflow.tracking.trajectory import TrajectoryBuilder


class TestDatabaseManager:
    """Tests for database manager."""
    
    def setup_method(self):
        """Set up test database."""
        self.db = DatabaseManager(":memory:")
    
    def teardown_method(self):
        """Clean up test database."""
        self.db.close()
    
    def test_insert_detection(self):
        """Test inserting a detection event."""
        event = DetectionEvent(
            token="abcd1234efgh5678",
            timestamp=datetime.now(),
            station_id="station_001",
            camera_id="camera_001",
            confidence=0.95,
            direction="entry"
        )
        
        row_id = self.db.insert_detection(event)
        
        assert row_id > 0
    
    def test_get_detections(self):
        """Test retrieving detection events."""
        # Insert multiple events
        for i in range(5):
            event = DetectionEvent(
                token=f"token_{i:016x}",
                timestamp=datetime.now(),
                station_id="station_001",
                camera_id="camera_001",
                confidence=0.9
            )
            self.db.insert_detection(event)
        
        # Retrieve all
        events = self.db.get_detections()
        
        assert len(events) == 5
    
    def test_get_detections_by_station(self):
        """Test filtering detections by station."""
        # Insert events for different stations
        for station in ["A", "B", "A", "C", "A"]:
            event = DetectionEvent(
                token="token_12345678",
                timestamp=datetime.now(),
                station_id=station,
                camera_id="camera_001",
                confidence=0.9
            )
            self.db.insert_detection(event)
        
        # Get only station A
        events = self.db.get_detections(station_id="A")
        
        assert len(events) == 3
        assert all(e.station_id == "A" for e in events)
    
    def test_get_unique_token_count(self):
        """Test counting unique tokens."""
        # Insert events with some duplicate tokens
        tokens = ["token1", "token2", "token1", "token3", "token2"]
        for token in tokens:
            event = DetectionEvent(
                token=token.ljust(16, '0'),
                timestamp=datetime.now(),
                station_id="station_001",
                camera_id="camera_001",
                confidence=0.9
            )
            self.db.insert_detection(event)
        
        count = self.db.get_unique_token_count("station_001")
        
        assert count == 3  # token1, token2, token3
    
    def test_purge_detections(self):
        """Test purging all detection data."""
        # Insert some events
        for i in range(10):
            event = DetectionEvent(
                token=f"token_{i:016x}",
                timestamp=datetime.now(),
                station_id="station_001",
                camera_id="camera_001",
                confidence=0.9
            )
            self.db.insert_detection(event)
        
        # Verify events exist
        assert len(self.db.get_detections()) == 10
        
        # Purge
        deleted = self.db.purge_detections()
        
        assert deleted == 10
        assert len(self.db.get_detections()) == 0


class TestDailyPurge:
    """Tests for daily purge mechanism."""
    
    def setup_method(self):
        """Set up test components."""
        self.db = DatabaseManager(":memory:")
        self.token_generator = PrivacyTokenGenerator()
        self.trajectory_builder = TrajectoryBuilder()
    
    def teardown_method(self):
        """Clean up."""
        self.db.close()
    
    def test_purge_executes(self):
        """Test that purge executes successfully."""
        purge = DailyPurge(
            db=self.db,
            token_generator=self.token_generator,
            trajectory_builder=self.trajectory_builder
        )
        
        # Add some data
        for i in range(5):
            event = DetectionEvent(
                token=f"token_{i:016x}",
                timestamp=datetime.now(),
                station_id="station_001",
                camera_id="camera_001",
                confidence=0.9
            )
            self.db.insert_detection(event)
            self.trajectory_builder.add_detection(event)
        
        # Execute purge
        stats = purge.execute()
        
        assert stats['status'] == 'success'
        assert stats['records_deleted'] == 5
        assert stats['new_salt_generated'] == True
        assert stats['trajectory_cleared'] == True
    
    def test_purge_clears_database(self):
        """Test that purge clears all detection data."""
        purge = DailyPurge(db=self.db)
        
        # Add data
        for i in range(10):
            event = DetectionEvent(
                token=f"token_{i:016x}",
                timestamp=datetime.now(),
                station_id="station_001",
                camera_id="camera_001",
                confidence=0.9
            )
            self.db.insert_detection(event)
        
        # Execute purge
        purge.execute()
        
        # Verify data is cleared
        assert len(self.db.get_detections()) == 0
    
    def test_purge_rotates_salt(self):
        """Test that purge rotates the token salt."""
        purge = DailyPurge(
            db=self.db,
            token_generator=self.token_generator
        )
        
        # Get salt before purge
        salt_before = self.token_generator.daily_salt
        
        # Execute purge
        purge.execute()
        
        # Get salt after purge
        salt_after = self.token_generator.daily_salt
        
        # Salt should have changed
        assert salt_before != salt_after
    
    def test_purge_clears_trajectories(self):
        """Test that purge clears trajectory data."""
        purge = DailyPurge(
            db=self.db,
            trajectory_builder=self.trajectory_builder
        )
        
        # Add trajectory data
        for i in range(5):
            event = DetectionEvent(
                token=f"token_{i:016x}",
                timestamp=datetime.now(),
                station_id="station_001",
                camera_id="camera_001",
                confidence=0.9
            )
            self.trajectory_builder.add_detection(event)
        
        assert self.trajectory_builder.event_count > 0
        
        # Execute purge
        purge.execute()
        
        # Verify trajectories are cleared
        assert self.trajectory_builder.event_count == 0
        assert self.trajectory_builder.trajectory_count == 0
    
    def test_purge_failure_calls_handler(self):
        """Test that purge failure calls the failure handler."""
        failure_called = []
        
        def on_failure(error):
            failure_called.append(error)
        
        # Create purge with broken database
        broken_db = DatabaseManager(":memory:")
        broken_db.close()  # Close to cause failure
        
        purge = DailyPurge(
            db=broken_db,
            on_failure=on_failure
        )
        
        # Execute purge (should fail)
        try:
            purge.execute()
        except Exception:
            pass  # Expected
        
        # Failure handler should have been called
        assert len(failure_called) > 0


class TestTrajectoryBuilder:
    """Tests for trajectory builder."""
    
    def setup_method(self):
        """Set up test trajectory builder."""
        self.builder = TrajectoryBuilder()
    
    def test_add_detection(self):
        """Test adding detection events."""
        event = DetectionEvent(
            token="token_12345678",
            timestamp=datetime.now(),
            station_id="station_A",
            camera_id="camera_001",
            confidence=0.9
        )
        
        self.builder.add_detection(event)
        
        assert self.builder.event_count == 1
        assert self.builder.trajectory_count == 1
    
    def test_trajectory_building(self):
        """Test that trajectories are built from events."""
        token = "token_12345678"
        
        # Add multiple events for same token
        for i, station in enumerate(["A", "B", "C"]):
            event = DetectionEvent(
                token=token,
                timestamp=datetime.now() + timedelta(minutes=i * 5),
                station_id=station,
                camera_id="camera_001",
                confidence=0.9
            )
            self.builder.add_detection(event)
        
        trajectories = self.builder.get_trajectories(min_length=2)
        
        assert len(trajectories) == 1
        assert len(trajectories[0].events) == 3
        assert trajectories[0].stations_visited == ["A", "B", "C"]
    
    def test_flow_matrix(self):
        """Test flow matrix generation."""
        # Create trajectory A -> B -> C
        token = "token_12345678"
        for i, station in enumerate(["A", "B", "C"]):
            event = DetectionEvent(
                token=token,
                timestamp=datetime.now() + timedelta(minutes=i * 5),
                station_id=station,
                camera_id="camera_001",
                confidence=0.9
            )
            self.builder.add_detection(event)
        
        matrix = self.builder.get_flow_matrix(["A", "B", "C"])
        
        # A -> B should be 1
        assert matrix[0, 1] == 1
        # B -> C should be 1
        assert matrix[1, 2] == 1
    
    def test_clear(self):
        """Test clearing all trajectory data."""
        # Add some events
        for i in range(10):
            event = DetectionEvent(
                token=f"token_{i:016x}",
                timestamp=datetime.now(),
                station_id="station_001",
                camera_id="camera_001",
                confidence=0.9
            )
            self.builder.add_detection(event)
        
        assert self.builder.event_count == 10
        
        # Clear
        deleted = self.builder.clear()
        
        assert deleted == 10
        assert self.builder.event_count == 0
        assert self.builder.trajectory_count == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
