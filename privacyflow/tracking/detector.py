"""
Person Detection and Feature Extraction

Handles person detection and extraction of all features
needed for privacy-preserving tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from privacyflow.features.face import FaceFeatureExtractor, FACE_EMBEDDING_DIM
from privacyflow.features.body import BodyProportionExtractor, BODY_FEATURE_DIM
from privacyflow.features.appearance import AppearanceExtractor, APPEARANCE_FEATURE_DIM


@dataclass
class Detection:
    """Single person detection result."""
    
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    pose_keypoints: Optional[np.ndarray] = None
    face_embedding: Optional[np.ndarray] = None
    body_proportions: Optional[np.ndarray] = None
    appearance_features: Optional[np.ndarray] = None
    face_crop: Optional[np.ndarray] = None
    
    @property
    def has_all_features(self) -> bool:
        """Check if all features have been extracted."""
        return (
            self.face_embedding is not None and
            self.body_proportions is not None and
            self.appearance_features is not None
        )


@dataclass
class FeatureVector:
    """Combined feature vector for a person."""
    
    face_embedding: np.ndarray = field(default_factory=lambda: np.zeros(FACE_EMBEDDING_DIM))
    body_proportions: np.ndarray = field(default_factory=lambda: np.zeros(BODY_FEATURE_DIM))
    appearance_features: np.ndarray = field(default_factory=lambda: np.zeros(APPEARANCE_FEATURE_DIM))
    
    @property
    def combined(self) -> np.ndarray:
        """Get concatenated feature vector (104 dimensions)."""
        return np.concatenate([
            self.face_embedding,
            self.body_proportions,
            self.appearance_features
        ])


class PersonDetector:
    """
    Handles person detection and feature extraction.
    
    Combines multiple detection and feature extraction models
    to generate privacy-preserving person representations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize person detector with configuration.
        
        Args:
            config: Configuration dictionary with model paths and thresholds
        """
        self.config = config or {}
        
        # Detection settings
        self.person_confidence_threshold = self.config.get('person_confidence_threshold', 0.6)
        self.face_confidence_threshold = self.config.get('face_confidence_threshold', 0.5)
        self.min_detection_size = self.config.get('min_detection_size', (50, 100))
        
        # Initialize feature extractors
        self.face_extractor = FaceFeatureExtractor(
            model_path=self.config.get('face_model')
        )
        self.body_extractor = BodyProportionExtractor(
            model_path=self.config.get('pose_model')
        )
        self.appearance_extractor = AppearanceExtractor(
            model_path=self.config.get('appearance_model'),
            color_bins=self.config.get('color_quantization_bins', 8)
        )
        
        # Person detection model (YOLO or similar)
        self.detector_model = None
        self._load_detector()
    
    def _load_detector(self) -> None:
        """Load person detection model."""
        model_path = self.config.get('person_detector_model')
        if model_path is not None:
            try:
                import onnxruntime as ort
                self.detector_model = ort.InferenceSession(model_path)
            except (ImportError, FileNotFoundError):
                self.detector_model = None
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in frame.
        
        Args:
            frame: Input image (BGR format)
        
        Returns:
            List of Detection objects with bounding boxes
        """
        if self.detector_model is not None:
            return self._detect_with_model(frame)
        else:
            return self._detect_simple(frame)
    
    def _detect_with_model(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons using ONNX model."""
        # Prepare input
        h, w = frame.shape[:2]
        input_size = (640, 480)  # Standard input size
        
        # Resize and normalize
        try:
            import cv2
            resized = cv2.resize(frame, input_size)
        except ImportError:
            # Simple resize fallback
            resized = frame
        
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        try:
            outputs = self.detector_model.run(None, {"images": input_tensor})
            return self._parse_detections(outputs[0], (w, h), input_size)
        except Exception:
            return []
    
    def _parse_detections(self, 
                          raw_output: np.ndarray,
                          original_size: Tuple[int, int],
                          input_size: Tuple[int, int]) -> List[Detection]:
        """Parse raw model output to Detection objects."""
        detections = []
        scale_x = original_size[0] / input_size[0]
        scale_y = original_size[1] / input_size[1]
        
        for det in raw_output:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = det[:6]
                
                # Filter by confidence and class (person = 0)
                if conf >= self.person_confidence_threshold and int(cls) == 0:
                    # Scale to original size
                    x = int(x1 * scale_x)
                    y = int(y1 * scale_y)
                    w = int((x2 - x1) * scale_x)
                    h = int((y2 - y1) * scale_y)
                    
                    # Filter by size
                    if w >= self.min_detection_size[0] and h >= self.min_detection_size[1]:
                        detections.append(Detection(
                            bbox=(x, y, w, h),
                            confidence=float(conf)
                        ))
        
        return detections
    
    def _detect_simple(self, frame: np.ndarray) -> List[Detection]:
        """
        Simple fallback detection using background subtraction.
        
        This is a placeholder for when no model is available.
        In production, a proper person detector should be used.
        """
        # This is a very basic placeholder
        # Returns empty list as we need a real detector
        return []
    
    def extract_features(self, frame: np.ndarray, detection: Detection) -> FeatureVector:
        """
        Extract all features for a detection.
        
        Args:
            frame: Input image
            detection: Detection object with bounding box
        
        Returns:
            FeatureVector containing all extracted features
        """
        features = FeatureVector()
        
        # Extract face features
        if detection.face_bbox is not None:
            features.face_embedding = self.face_extractor.extract(frame, detection.face_bbox)
        else:
            # Estimate face location from person bbox
            face_bbox = self._estimate_face_bbox(detection.bbox)
            features.face_embedding = self.face_extractor.extract(frame, face_bbox)
            detection.face_bbox = face_bbox
        
        # Store face crop for validation (32x32 resolution)
        detection.face_crop = self._get_face_crop(frame, detection.face_bbox)
        
        # Extract body proportions
        if detection.pose_keypoints is not None:
            features.body_proportions = self.body_extractor.extract(detection.pose_keypoints)
        else:
            features.body_proportions = self.body_extractor.extract_from_bbox(frame, detection.bbox)
        
        # Extract appearance features
        features.appearance_features = self.appearance_extractor.extract(
            frame, detection.bbox, None
        )
        
        # Store features in detection
        detection.face_embedding = features.face_embedding
        detection.body_proportions = features.body_proportions
        detection.appearance_features = features.appearance_features
        
        return features
    
    def _estimate_face_bbox(self, person_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Estimate face bounding box from person bounding box.
        
        Uses typical human proportions to estimate face location.
        """
        x, y, w, h = person_bbox
        
        # Face is typically in upper 1/5 of body, centered horizontally
        face_h = int(h * 0.15)
        face_w = int(w * 0.4)
        face_x = x + (w - face_w) // 2
        face_y = y + int(h * 0.02)  # Slightly below top
        
        return (face_x, face_y, face_w, face_h)
    
    def _get_face_crop(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Get face crop at privacy-preserving resolution (32x32).
        
        This is used to verify the privacy guarantee.
        """
        x, y, w, h = face_bbox
        x = max(0, x)
        y = max(0, y)
        
        h_frame, w_frame = frame.shape[:2]
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w <= 0 or h <= 0:
            return np.zeros((32, 32, 3), dtype=np.uint8)
        
        face_crop = frame[y:y+h, x:x+w]
        
        # Resize to 32x32 (privacy-preserving resolution)
        try:
            import cv2
            return cv2.resize(face_crop, (32, 32), interpolation=cv2.INTER_AREA)
        except ImportError:
            # Simple fallback resize
            target_h, target_w = 32, 32
            h_curr, w_curr = face_crop.shape[:2]
            row_indices = np.linspace(0, h_curr - 1, target_h).astype(int)
            col_indices = np.linspace(0, w_curr - 1, target_w).astype(int)
            if len(face_crop.shape) == 3:
                return face_crop[row_indices][:, col_indices, :]
            else:
                result = face_crop[row_indices][:, col_indices]
                return np.stack([result, result, result], axis=-1)
