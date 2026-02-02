"""
Tests for feature extraction modules.
"""

import pytest
import numpy as np

from privacyflow.features.face import FaceFeatureExtractor, FACE_RESOLUTION, FACE_EMBEDDING_DIM
from privacyflow.features.body import BodyProportionExtractor, BODY_FEATURE_DIM
from privacyflow.features.appearance import AppearanceExtractor, APPEARANCE_FEATURE_DIM


class TestFaceFeatureExtractor:
    """Tests for face feature extraction."""
    
    def setup_method(self):
        """Set up extractor."""
        self.extractor = FaceFeatureExtractor()
    
    def test_resolution_is_32x32(self):
        """Verify face resolution constant is 32x32 (privacy requirement)."""
        assert FACE_RESOLUTION == (32, 32)
    
    def test_embedding_dimension(self):
        """Verify embedding dimension is 64."""
        assert FACE_EMBEDDING_DIM == 64
    
    def test_extract_returns_correct_dim(self):
        """Test that extraction returns correct dimension."""
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 100, 100)
        
        embedding = self.extractor.extract(image, bbox)
        
        assert embedding.shape == (FACE_EMBEDDING_DIM,)
    
    def test_extract_with_invalid_bbox(self):
        """Test extraction with invalid bounding box."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Invalid bbox (outside image)
        bbox = (1000, 1000, 100, 100)
        embedding = self.extractor.extract(image, bbox)
        
        # Should return zeros
        assert embedding.shape == (FACE_EMBEDDING_DIM,)
        np.testing.assert_array_equal(embedding, np.zeros(FACE_EMBEDDING_DIM))
    
    def test_extract_with_zero_bbox(self):
        """Test extraction with zero-size bounding box."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        bbox = (100, 100, 0, 0)
        embedding = self.extractor.extract(image, bbox)
        
        assert embedding.shape == (FACE_EMBEDDING_DIM,)
    
    def test_embedding_is_normalized(self):
        """Test that embedding is L2 normalized."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 100, 100)
        
        embedding = self.extractor.extract(image, bbox)
        
        # L2 norm should be 1 (or 0 for zero vectors)
        norm = np.linalg.norm(embedding)
        assert norm == pytest.approx(1.0, abs=0.01) or norm == 0


class TestBodyProportionExtractor:
    """Tests for body proportion extraction."""
    
    def setup_method(self):
        """Set up extractor."""
        self.extractor = BodyProportionExtractor()
    
    def test_feature_dimension(self):
        """Verify body feature dimension is 8."""
        assert BODY_FEATURE_DIM == 8
    
    def test_extract_with_keypoints(self):
        """Test extraction with pose keypoints."""
        # Create 17-point pose (COCO format)
        keypoints = np.random.rand(17, 3).astype(np.float32)
        keypoints[:, :2] *= 200  # Scale positions
        keypoints[:, 2] = 0.9  # High confidence
        
        proportions = self.extractor.extract(keypoints)
        
        assert proportions.shape == (BODY_FEATURE_DIM,)
    
    def test_extract_from_bbox(self):
        """Test fallback extraction from bounding box."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 50, 100, 300)  # Typical person bbox
        
        proportions = self.extractor.extract_from_bbox(image, bbox)
        
        assert proportions.shape == (BODY_FEATURE_DIM,)
    
    def test_proportions_are_ratios(self):
        """Verify all proportions are in reasonable ratio range."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        keypoints[:, :2] *= 200
        keypoints[:, 2] = 0.9
        
        proportions = self.extractor.extract(keypoints)
        
        # All ratios should be between 0 and some reasonable max
        # (normalized vector, so values should be reasonable)
        assert proportions.min() >= -1.0
        assert proportions.max() <= 1.0
    
    def test_extract_with_none_keypoints(self):
        """Test extraction with None keypoints."""
        proportions = self.extractor.extract(None)
        
        assert proportions.shape == (BODY_FEATURE_DIM,)
        np.testing.assert_array_equal(proportions, np.zeros(BODY_FEATURE_DIM))


class TestAppearanceExtractor:
    """Tests for appearance feature extraction."""
    
    def setup_method(self):
        """Set up extractor."""
        self.extractor = AppearanceExtractor()
    
    def test_feature_dimension(self):
        """Verify appearance feature dimension is 32."""
        assert APPEARANCE_FEATURE_DIM == 32
    
    def test_extract_returns_correct_dim(self):
        """Test that extraction returns correct dimension."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 50, 150, 400)
        
        features = self.extractor.extract(image, bbox)
        
        assert features.shape == (APPEARANCE_FEATURE_DIM,)
    
    def test_extract_with_invalid_bbox(self):
        """Test extraction with invalid bounding box."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        bbox = (1000, 1000, 100, 100)
        features = self.extractor.extract(image, bbox)
        
        assert features.shape == (APPEARANCE_FEATURE_DIM,)
    
    def test_extract_with_mask(self):
        """Test extraction with segmentation mask."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 50, 150, 400)
        mask = np.ones((400, 150), dtype=np.uint8)
        
        features = self.extractor.extract(image, bbox, mask)
        
        assert features.shape == (APPEARANCE_FEATURE_DIM,)
    
    def test_features_are_normalized(self):
        """Test that features are L2 normalized."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 50, 150, 400)
        
        features = self.extractor.extract(image, bbox)
        
        norm = np.linalg.norm(features)
        assert norm == pytest.approx(1.0, abs=0.01) or norm == 0


class TestFeatureCombination:
    """Tests for combining all features."""
    
    def test_total_feature_dimension(self):
        """Verify total combined feature dimension is 104."""
        total_dim = FACE_EMBEDDING_DIM + BODY_FEATURE_DIM + APPEARANCE_FEATURE_DIM
        assert total_dim == 104  # 64 + 8 + 32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
