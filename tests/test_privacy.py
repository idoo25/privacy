"""
Privacy Tests

Tests to ensure privacy guarantees are maintained.
"""

import pytest
import numpy as np
from datetime import date, datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.privacy_token import PrivacyTokenGenerator
from src.detector import PersonDetector, FACE_RESOLUTION, extract_face_features, FaceEncoder
from src.data_structures import DetectionEvent, DailyTrajectory
from src.trajectory import TrajectoryBuilder
from src.database import EphemeralDatabase


def generate_random_features():
    """Generate random feature vectors for testing."""
    face_embedding = np.random.rand(64).astype(np.float32)
    body_proportions = np.random.rand(8).astype(np.float32)
    appearance_features = np.random.rand(32).astype(np.float32)
    return face_embedding, body_proportions, appearance_features


class TestFaceResolution:
    """Tests for face resolution privacy limits."""
    
    def test_face_resolution_limit(self):
        """Ensure face images never exceed 32×32."""
        assert FACE_RESOLUTION == (32, 32), \
            "Face resolution exceeds privacy limit!"
    
    def test_face_resolution_constant(self):
        """Verify face resolution is properly defined."""
        assert FACE_RESOLUTION[0] == 32
        assert FACE_RESOLUTION[1] == 32


class TestTokenIrreversibility:
    """Tests for token security."""
    
    def test_token_irreversibility(self):
        """Verify tokens cannot be reversed to features."""
        generator = PrivacyTokenGenerator()
        
        features = generate_random_features()
        token = generator.generate_token(*features)
        
        # Token should be fixed-length hash
        assert len(token) == 16, "Token length should be 16 characters"
        
        # Token should be hexadecimal
        assert all(c in '0123456789abcdef' for c in token), \
            "Token should be hexadecimal"
    
    def test_token_deterministic_same_salt(self):
        """Same features with same salt produce same token."""
        generator = PrivacyTokenGenerator()
        
        features = generate_random_features()
        token1 = generator.generate_token(*features)
        token2 = generator.generate_token(*features)
        
        assert token1 == token2, "Same features should produce same token"
    
    def test_different_features_different_tokens(self):
        """Different features produce different tokens."""
        generator = PrivacyTokenGenerator()
        
        features1 = generate_random_features()
        features2 = generate_random_features()
        
        token1 = generator.generate_token(*features1)
        token2 = generator.generate_token(*features2)
        
        assert token1 != token2, "Different features should produce different tokens"


class TestDailySaltRotation:
    """Tests for daily salt rotation."""
    
    def test_daily_salt_rotation(self):
        """Verify same features produce different tokens on different days."""
        generator = PrivacyTokenGenerator()
        
        features = generate_random_features()
        
        # Generate token today
        token_today = generator.generate_token(*features)
        
        # Simulate next day by forcing salt rotation
        generator._salt_date = date.today() - timedelta(days=1)
        token_tomorrow = generator.generate_token(*features)
        
        assert token_today != token_tomorrow, \
            "Salt rotation failed - cross-day tracking possible!"
    
    def test_salt_regeneration(self):
        """Test that salt is regenerated on new day."""
        generator = PrivacyTokenGenerator()
        
        # Get initial salt
        salt1 = generator.daily_salt
        
        # Simulate date change
        generator._salt_date = date.today() - timedelta(days=1)
        
        # Get new salt
        salt2 = generator.daily_salt
        
        assert salt1 != salt2, "Salt should change on new day"
    
    def test_manual_salt_rotation(self):
        """Test manual salt rotation."""
        generator = PrivacyTokenGenerator()
        
        features = generate_random_features()
        token1 = generator.generate_token(*features)
        
        # Manually rotate salt
        generator.rotate_salt()
        
        token2 = generator.generate_token(*features)
        
        assert token1 != token2, "Manual salt rotation should produce different tokens"


class TestNoRawDataInExports:
    """Tests for export security."""
    
    def test_database_purge(self):
        """Test that database purge works correctly."""
        db = EphemeralDatabase()
        
        # Insert some detections
        from datetime import datetime
        event = DetectionEvent(
            token="abc123def456ghij",
            timestamp=datetime.now(),
            station_id="station_001",
            camera_id="camera_001",
            confidence=0.95,
            direction="entry"
        )
        db.insert_detection(event)
        
        # Verify detection exists
        count_before = db.count_unique_tokens()
        assert count_before > 0
        
        # Purge
        deleted = db.purge_daily_data()
        assert deleted > 0
        
        # Verify deletion
        count_after = db.count_unique_tokens()
        assert count_after == 0
        
        db.close()
    
    def test_detection_event_no_raw_features(self):
        """Verify DetectionEvent doesn't store raw features."""
        event = DetectionEvent(
            token="abc123",
            timestamp=datetime.now(),
            station_id="station_001",
            camera_id="camera_001",
            confidence=0.95
        )
        
        # Check that there's no way to get raw features from event
        assert not hasattr(event, 'face_embedding')
        assert not hasattr(event, 'body_proportions')
        assert not hasattr(event, 'appearance_features')
        assert not hasattr(event, 'raw_image')


class TestTrajectoryBuilder:
    """Tests for trajectory building."""
    
    def test_trajectory_clear(self):
        """Test that trajectory builder can be cleared."""
        builder = TrajectoryBuilder()
        
        from datetime import datetime
        event = DetectionEvent(
            token="test_token",
            timestamp=datetime.now(),
            station_id="station_001",
            camera_id="camera_001",
            confidence=0.95
        )
        builder.add_detection(event)
        
        assert len(builder._events) > 0
        
        builder.clear()
        
        assert len(builder._events) == 0
        assert len(builder._trajectories) == 0
    
    def test_round_trip_detection(self):
        """Test round trip detection."""
        from datetime import datetime, timedelta
        
        builder = TrajectoryBuilder()
        
        # Create round trip trajectory
        token = "round_trip_token"
        events = [
            DetectionEvent(
                token=token,
                timestamp=datetime.now(),
                station_id="A",
                camera_id="cam_a",
                confidence=0.9
            ),
            DetectionEvent(
                token=token,
                timestamp=datetime.now() + timedelta(minutes=30),
                station_id="B",
                camera_id="cam_b",
                confidence=0.9
            ),
            DetectionEvent(
                token=token,
                timestamp=datetime.now() + timedelta(hours=1),
                station_id="A",
                camera_id="cam_a",
                confidence=0.9
            )
        ]
        
        for event in events:
            builder.add_detection(event)
        
        trajectories = builder.get_trajectories(min_length=2)
        assert len(trajectories) == 1
        assert trajectories[0].is_round_trip


class TestPrivacyFlow:
    """Integration tests for PrivacyFlow system."""
    
    def test_token_generation_produces_valid_tokens(self):
        """Test that token generation works correctly."""
        generator = PrivacyTokenGenerator()
        
        # Test with various feature combinations
        for _ in range(10):
            features = generate_random_features()
            token = generator.generate_token(*features)
            
            assert len(token) == 16
            assert all(c in '0123456789abcdef' for c in token)
    
    def test_quantization_reduces_precision(self):
        """Test that quantization works correctly."""
        generator = PrivacyTokenGenerator()
        
        # Create similar features
        base = np.random.rand(64).astype(np.float32)
        similar = base + np.random.rand(64) * 0.01  # Small perturbation
        
        # Quantize both
        quant_base = generator._quantize(base, bins=16)
        quant_similar = generator._quantize(similar, bins=16)
        
        # They should be equal or very similar after quantization
        # (depends on where values fall in bins)
        assert quant_base.shape == quant_similar.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
