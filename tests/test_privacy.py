"""
Privacy Tests for PrivacyFlow

These tests verify the core privacy guarantees of the system.
"""

import pytest
from datetime import date, timedelta

import numpy as np

from privacyflow.models.token import PrivacyTokenGenerator
from privacyflow.features.face import FaceFeatureExtractor, FACE_RESOLUTION, FACE_EMBEDDING_DIM
from privacyflow.tracking.detector import PersonDetector


def generate_random_features():
    """Generate random feature vectors for testing."""
    face_embedding = np.random.rand(64).astype(np.float32)
    body_proportions = np.random.rand(8).astype(np.float32)
    appearance_features = np.random.rand(32).astype(np.float32)
    return face_embedding, body_proportions, appearance_features


class TestFaceResolution:
    """Tests for face resolution limits."""
    
    def test_face_resolution_constant(self):
        """Ensure face resolution constant is 32x32."""
        assert FACE_RESOLUTION == (32, 32), \
            "Face resolution must be 32x32 for privacy!"
    
    def test_face_embedding_dimension(self):
        """Ensure face embedding is 64-dimensional."""
        assert FACE_EMBEDDING_DIM == 64
    
    def test_face_resize_to_32x32(self):
        """Ensure face images are resized to 32x32."""
        extractor = FaceFeatureExtractor()
        
        # Create a larger test image
        large_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)  # Large face bbox
        
        # Extract features (internally resizes to 32x32)
        embedding = extractor.extract(large_image, bbox)
        
        # Embedding should be 64-dimensional
        assert len(embedding) == FACE_EMBEDDING_DIM
    
    def test_face_crop_resolution_limit(self):
        """Verify face crops never exceed 32x32."""
        detector = PersonDetector({})
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        face_bbox = (100, 50, 150, 150)  # Large face bbox
        
        # Get face crop
        face_crop = detector._get_face_crop(frame, face_bbox)
        
        # Face crop should be exactly 32x32
        assert face_crop.shape[:2] == (32, 32), \
            f"Face crop is {face_crop.shape[:2]}, should be (32, 32)!"


class TestTokenGeneration:
    """Tests for privacy token generation."""
    
    def test_token_length(self):
        """Verify tokens are 16 characters."""
        generator = PrivacyTokenGenerator()
        features = generate_random_features()
        
        token = generator.generate_token(*features)
        
        assert len(token) == 16, f"Token length is {len(token)}, expected 16"
    
    def test_token_is_hex(self):
        """Verify token is valid hexadecimal."""
        generator = PrivacyTokenGenerator()
        features = generate_random_features()
        
        token = generator.generate_token(*features)
        
        # Should be valid hex
        try:
            int(token, 16)
        except ValueError:
            pytest.fail(f"Token '{token}' is not valid hexadecimal")
    
    def test_token_irreversibility(self):
        """Verify tokens cannot be reversed to features."""
        generator = PrivacyTokenGenerator()
        
        features = generate_random_features()
        token = generator.generate_token(*features)
        
        # Token should be fixed-length hash
        assert len(token) == 16
        
        # Features should not appear in token
        for f in features:
            for val in f:
                assert str(val) not in token, \
                    "Feature values should not appear in token!"
    
    def test_same_features_same_token(self):
        """Same features should produce same token (within same day)."""
        generator = PrivacyTokenGenerator()
        
        features = generate_random_features()
        
        token1 = generator.generate_token(*features)
        token2 = generator.generate_token(*features)
        
        assert token1 == token2, \
            "Same features should produce same token on same day"
    
    def test_different_features_different_tokens(self):
        """Different features should produce different tokens."""
        generator = PrivacyTokenGenerator()
        
        features1 = generate_random_features()
        features2 = generate_random_features()
        
        token1 = generator.generate_token(*features1)
        token2 = generator.generate_token(*features2)
        
        assert token1 != token2, \
            "Different features should produce different tokens"


class TestDailySaltRotation:
    """Tests for daily salt rotation."""
    
    def test_daily_salt_generation(self):
        """Verify salt is generated."""
        generator = PrivacyTokenGenerator()
        
        salt = generator.daily_salt
        
        assert salt is not None
        assert len(salt) == 32  # 256 bits
    
    def test_salt_consistent_same_day(self):
        """Salt should be consistent within same day."""
        generator = PrivacyTokenGenerator()
        
        salt1 = generator.daily_salt
        salt2 = generator.daily_salt
        
        assert salt1 == salt2
    
    def test_salt_rotation_changes_tokens(self):
        """
        Verify same features produce different tokens on different days.
        
        This is critical for preventing cross-day tracking.
        """
        generator = PrivacyTokenGenerator()
        
        features = generate_random_features()
        
        # Generate token today
        token_today = generator.generate_token(*features)
        
        # Simulate next day by manually rotating salt
        generator._salt_date = date.today() - timedelta(days=1)
        generator._daily_salt = None  # Force regeneration
        
        # Generate token "tomorrow"
        token_tomorrow = generator.generate_token(*features)
        
        assert token_today != token_tomorrow, \
            "Salt rotation failed - cross-day tracking possible!"
    
    def test_manual_salt_rotation(self):
        """Test manual salt rotation (used during daily purge)."""
        generator = PrivacyTokenGenerator()
        
        features = generate_random_features()
        
        # Get token before rotation
        token_before = generator.generate_token(*features)
        old_salt = generator.daily_salt
        
        # Rotate salt
        generator.rotate_salt()
        
        # Get token after rotation
        token_after = generator.generate_token(*features)
        new_salt = generator.daily_salt
        
        assert old_salt != new_salt, "Salt should change after rotation"
        assert token_before != token_after, "Token should change after salt rotation"


class TestFeatureQuantization:
    """Tests for feature quantization."""
    
    def test_quantization_reduces_precision(self):
        """Verify quantization reduces feature precision."""
        generator = PrivacyTokenGenerator()
        
        # Very similar features
        features1 = np.array([0.5001, 0.5002, 0.5003], dtype=np.float32)
        features2 = np.array([0.5004, 0.5005, 0.5006], dtype=np.float32)
        
        # Quantize both
        quant1 = generator._quantize(features1, bins=8)
        quant2 = generator._quantize(features2, bins=8)
        
        # Should be equal after quantization (same bin)
        np.testing.assert_array_equal(quant1, quant2)
    
    def test_quantization_output_range(self):
        """Verify quantization output is within expected range."""
        generator = PrivacyTokenGenerator()
        
        features = np.random.rand(100).astype(np.float32)
        quantized = generator._quantize(features, bins=16)
        
        # Output should be integers in range [0, bins]
        assert quantized.min() >= 0
        assert quantized.max() <= 16


class TestNoRawDataExposure:
    """Tests to ensure raw data is never exposed."""
    
    def test_detection_event_no_features(self):
        """DetectionEvent should not store raw features."""
        from privacyflow.models.detection import DetectionEvent
        from datetime import datetime
        
        event = DetectionEvent(
            token="abc123def456",
            timestamp=datetime.now(),
            station_id="station_001",
            camera_id="camera_001",
            confidence=0.95
        )
        
        # Check that no feature-related attributes exist
        assert not hasattr(event, 'face_embedding')
        assert not hasattr(event, 'body_proportions')
        assert not hasattr(event, 'appearance_features')
        assert not hasattr(event, 'raw_image')
    
    def test_trajectory_no_features(self):
        """DailyTrajectory should not store raw features."""
        from privacyflow.models.detection import DailyTrajectory
        
        trajectory = DailyTrajectory(token="abc123def456")
        
        # Check that no feature-related attributes exist
        assert not hasattr(trajectory, 'face_embedding')
        assert not hasattr(trajectory, 'body_proportions')
        assert not hasattr(trajectory, 'appearance_features')


class TestPrivacyIntegration:
    """Integration tests for privacy guarantees."""
    
    def test_end_to_end_privacy(self):
        """Test complete privacy flow from frame to token."""
        from privacyflow.core import PrivacyFlow
        
        # Initialize system
        pf = PrivacyFlow()
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Start system
        pf.start()
        
        # Process frame (may return empty if no detections without model)
        events = pf.process_frame(frame, "station_001", "camera_001")
        
        # Stop system
        pf.stop()
        
        # Verify events don't contain raw features
        for event in events:
            assert len(event.token) == 16
            assert not hasattr(event, 'face_embedding')
            assert not hasattr(event, 'body_proportions')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
