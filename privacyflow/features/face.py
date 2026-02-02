"""
Face Feature Extraction - Ultra-Low Resolution (Privacy by Design)

The face resolution is INTENTIONALLY limited to 32x32 pixels.
This is a core privacy feature - DO NOT INCREASE.
"""

from typing import Tuple, Optional

import numpy as np

# Configuration - DO NOT INCREASE RESOLUTION
FACE_RESOLUTION: Tuple[int, int] = (32, 32)  # Intentionally low - privacy by design
FACE_EMBEDDING_DIM: int = 64


def normalize(vector: np.ndarray) -> np.ndarray:
    """L2 normalize a feature vector."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


class FaceFeatureExtractor:
    """
    Extract face features at privacy-preserving resolution.
    
    IMPORTANT: Face images are ALWAYS resized to 32x32 pixels.
    This resolution is too low for biometric identification,
    which is the core privacy guarantee of this system.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face feature extractor.
        
        Args:
            model_path: Path to ONNX face encoder model.
                       If None, uses a simple encoder.
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the face encoding model."""
        if self.model_path is not None:
            try:
                import onnxruntime as ort
                self.model = ort.InferenceSession(self.model_path)
            except (ImportError, FileNotFoundError):
                self.model = None
    
    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract face features at privacy-preserving resolution.
        
        Args:
            frame: Input image (any resolution)
            bbox: Face bounding box [x, y, w, h]
        
        Returns:
            numpy.ndarray: 64-dimensional embedding
        """
        x, y, w, h = bbox
        
        # Ensure valid bounding box
        x = max(0, x)
        y = max(0, y)
        h_frame, w_frame = frame.shape[:2]
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w <= 0 or h <= 0:
            return np.zeros(FACE_EMBEDDING_DIM)
        
        # Crop face region
        face_crop = frame[y:y+h, x:x+w]
        
        # CRITICAL: Resize to privacy-preserving resolution
        face_resized = self._resize_face(face_crop)
        
        # Encode face
        embedding = self._encode(face_resized)
        
        return normalize(embedding)
    
    def _resize_face(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Resize face to privacy-preserving resolution.
        
        This is the CORE privacy mechanism for face detection.
        32x32 pixels is too low for facial recognition systems.
        """
        try:
            import cv2
            return cv2.resize(face_crop, FACE_RESOLUTION, interpolation=cv2.INTER_AREA)
        except ImportError:
            # Fallback: simple resize using numpy
            h, w = face_crop.shape[:2]
            target_h, target_w = FACE_RESOLUTION
            
            # Calculate indices for downsampling
            row_indices = np.linspace(0, h - 1, target_h).astype(int)
            col_indices = np.linspace(0, w - 1, target_w).astype(int)
            
            if len(face_crop.shape) == 3:
                return face_crop[row_indices][:, col_indices, :]
            else:
                return face_crop[row_indices][:, col_indices]
    
    def _encode(self, face_image: np.ndarray) -> np.ndarray:
        """
        Encode face image to embedding vector.
        
        Args:
            face_image: 32x32 pixel face image
        
        Returns:
            64-dimensional embedding vector
        """
        if self.model is not None:
            # Use ONNX model if available
            input_tensor = face_image.astype(np.float32) / 255.0
            if len(input_tensor.shape) == 2:
                input_tensor = np.expand_dims(input_tensor, axis=(0, -1))
            elif len(input_tensor.shape) == 3:
                input_tensor = np.expand_dims(input_tensor, axis=0)
            
            try:
                outputs = self.model.run(None, {"input": input_tensor})
                return outputs[0].flatten()[:FACE_EMBEDDING_DIM]
            except Exception:
                pass
        
        # Fallback: Simple feature extraction based on image statistics
        return self._simple_encode(face_image)
    
    def _simple_encode(self, face_image: np.ndarray) -> np.ndarray:
        """
        Simple fallback encoding when model is unavailable.
        
        Uses basic image statistics as features.
        """
        # Normalize image
        if face_image.dtype == np.uint8:
            img = face_image.astype(np.float32) / 255.0
        else:
            img = face_image.astype(np.float32)
        
        features = []
        
        # Mean and std per channel
        if len(img.shape) == 3:
            for c in range(img.shape[2]):
                features.extend([img[:, :, c].mean(), img[:, :, c].std()])
        else:
            features.extend([img.mean(), img.std()])
        
        # Histogram features
        hist, _ = np.histogram(img.flatten(), bins=32, range=(0, 1))
        hist = hist.astype(np.float32) / hist.sum()
        features.extend(hist)
        
        # Quadrant features
        h, w = img.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        if len(img.shape) == 3:
            img_gray = img.mean(axis=2)
        else:
            img_gray = img
        
        quadrants = [
            img_gray[:mid_h, :mid_w],
            img_gray[:mid_h, mid_w:],
            img_gray[mid_h:, :mid_w],
            img_gray[mid_h:, mid_w:]
        ]
        for q in quadrants:
            features.extend([q.mean(), q.std()])
        
        # Pad or truncate to FACE_EMBEDDING_DIM
        features = np.array(features)
        if len(features) < FACE_EMBEDDING_DIM:
            features = np.pad(features, (0, FACE_EMBEDDING_DIM - len(features)))
        else:
            features = features[:FACE_EMBEDDING_DIM]
        
        return features
