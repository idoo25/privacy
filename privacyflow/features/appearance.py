"""
Appearance Feature Extraction

WARNING: These features are intentionally volatile.
Changing clothes = new identity (by design).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# Appearance categories
APPEARANCE_FEATURES: Dict[str, type] = {
    'upper_body_colors': List[Tuple[int, int, int]],    # Dominant colors (top 3)
    'lower_body_colors': List[Tuple[int, int, int]],    # Dominant colors (top 3)
    'footwear_type': int,              # Category index
    'headwear_present': bool,
    'bag_present': bool,
    'bag_position': int,               # Back/front/side
    'jacket_present': bool,
    'pattern_type': int                # Solid/striped/plaid/etc
}

APPEARANCE_FEATURE_DIM: int = 32
COLOR_QUANTIZATION_BINS: int = 8


def normalize(vector: np.ndarray) -> np.ndarray:
    """L2 normalize a feature vector."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


class AppearanceExtractor:
    """
    Extract appearance features for short-term tracking.
    
    WARNING: These features are intentionally volatile.
    Changing clothes = new identity (by design).
    """
    
    def __init__(self, model_path: Optional[str] = None, color_bins: int = COLOR_QUANTIZATION_BINS):
        """
        Initialize appearance extractor.
        
        Args:
            model_path: Path to ONNX appearance encoder model.
            color_bins: Number of bins for color quantization.
        """
        self.model_path = model_path
        self.color_bins = color_bins
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load appearance encoding model."""
        if self.model_path is not None:
            try:
                import onnxruntime as ort
                self.model = ort.InferenceSession(self.model_path)
            except (ImportError, FileNotFoundError):
                self.model = None
    
    def extract(self, 
                frame: np.ndarray,
                person_bbox: Tuple[int, int, int, int],
                segmentation_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract appearance features for short-term tracking.
        
        WARNING: These features are intentionally volatile.
        Changing clothes = new identity (by design).
        
        Args:
            frame: Input image
            person_bbox: Person bounding box (x, y, w, h)
            segmentation_mask: Body part segmentation (optional)
        
        Returns:
            numpy.ndarray: 32-dimensional appearance vector
        """
        x, y, w, h = person_bbox
        
        # Ensure valid bounding box
        x = max(0, x)
        y = max(0, y)
        h_frame, w_frame = frame.shape[:2]
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w <= 0 or h <= 0:
            return np.zeros(APPEARANCE_FEATURE_DIM)
        
        # Crop person region
        person_crop = frame[y:y+h, x:x+w]
        
        # Extract color features
        upper_colors = self._extract_dominant_colors(person_crop, 'upper', segmentation_mask)
        lower_colors = self._extract_dominant_colors(person_crop, 'lower', segmentation_mask)
        
        # Detect accessories
        accessories = self._detect_accessories(person_crop)
        
        # Encode all features
        return self._encode_appearance(upper_colors, lower_colors, accessories)
    
    def _extract_dominant_colors(self, 
                                  person_crop: np.ndarray,
                                  region: str,
                                  mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract dominant colors from a region.
        
        Args:
            person_crop: Cropped person image
            region: 'upper' or 'lower' body region
            mask: Optional segmentation mask
        
        Returns:
            Dominant color features
        """
        h, w = person_crop.shape[:2]
        
        # Define region based on typical body proportions
        if region == 'upper':
            region_crop = person_crop[:h//2, :]  # Upper half
        else:
            region_crop = person_crop[h//2:, :]  # Lower half
        
        if mask is not None:
            # Apply mask if available
            if region == 'upper':
                region_mask = mask[:h//2, :]
            else:
                region_mask = mask[h//2:, :]
            region_crop = region_crop[region_mask > 0]
        
        if region_crop.size == 0:
            return np.zeros(9)  # 3 colors * 3 RGB channels
        
        # Quantize colors
        colors = self._quantize_colors(region_crop)
        
        # Get dominant colors (top 3)
        return self._get_dominant_colors(colors, n_colors=3).flatten()
    
    def _quantize_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Quantize colors to reduce sensitivity to lighting.
        
        Args:
            image: Input image region
        
        Returns:
            Quantized color values
        """
        if len(image.shape) == 1:
            # Already flattened
            pixels = image.reshape(-1, 3) if image.size % 3 == 0 else image.reshape(-1, 1)
        else:
            pixels = image.reshape(-1, image.shape[-1]) if len(image.shape) > 2 else image.reshape(-1, 1)
        
        # Quantize to bins
        if pixels.max() > 1:
            pixels = pixels / 255.0
        
        quantized = (pixels * self.color_bins).astype(int)
        quantized = np.clip(quantized, 0, self.color_bins - 1)
        
        return quantized
    
    def _get_dominant_colors(self, quantized: np.ndarray, n_colors: int = 3) -> np.ndarray:
        """
        Get the n most dominant colors.
        
        Args:
            quantized: Quantized color values
            n_colors: Number of dominant colors to return
        
        Returns:
            Array of dominant colors (n_colors x channels)
        """
        if quantized.size == 0:
            return np.zeros((n_colors, 3))
        
        # Convert to tuples for counting
        if len(quantized.shape) == 1:
            quantized = quantized.reshape(-1, 1)
        
        n_channels = quantized.shape[1]
        
        # Simple histogram-based dominant color extraction
        if n_channels >= 3:
            # For RGB images, use simplified approach
            colors = quantized[:, :3]
            mean_color = colors.mean(axis=0)
            std_color = colors.std(axis=0)
            
            # Return mean and variations as "dominant" colors
            result = np.zeros((n_colors, 3))
            result[0] = mean_color
            result[1] = np.clip(mean_color + std_color, 0, self.color_bins - 1)
            result[2] = np.clip(mean_color - std_color, 0, self.color_bins - 1)
            
            # Normalize to [0, 1]
            return result / self.color_bins
        else:
            # Grayscale
            mean_val = quantized.mean()
            result = np.zeros((n_colors, 3))
            result[:, :] = mean_val / self.color_bins
            return result
    
    def _detect_accessories(self, person_crop: np.ndarray) -> Dict[str, any]:
        """
        Detect accessories in person crop.
        
        Args:
            person_crop: Cropped person image
        
        Returns:
            Dictionary of detected accessories
        """
        # Simple heuristic-based accessory detection
        # In production, this would use a trained model
        h, w = person_crop.shape[:2]
        
        accessories = {
            'footwear_type': 0,
            'headwear_present': False,
            'bag_present': False,
            'bag_position': 0,
            'jacket_present': False,
            'pattern_type': 0
        }
        
        # Check head region for headwear (top 20%)
        head_region = person_crop[:int(h * 0.2), :]
        if head_region.size > 0:
            head_variance = head_region.var()
            accessories['headwear_present'] = head_variance > 1000  # Threshold
        
        # Check for pattern by looking at edge variance
        try:
            import cv2
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY) if len(person_crop.shape) == 3 else person_crop
            edges = cv2.Canny(gray, 50, 150)
            edge_density = edges.sum() / edges.size
            if edge_density > 0.1:
                accessories['pattern_type'] = 1  # Patterned
            elif edge_density > 0.05:
                accessories['pattern_type'] = 2  # Striped
        except ImportError:
            pass
        
        return accessories
    
    def _encode_appearance(self,
                           upper_colors: np.ndarray,
                           lower_colors: np.ndarray,
                           accessories: Dict[str, any]) -> np.ndarray:
        """
        Encode appearance features into a fixed-size vector.
        
        Args:
            upper_colors: Upper body color features
            lower_colors: Lower body color features
            accessories: Detected accessories
        
        Returns:
            32-dimensional appearance vector
        """
        features = []
        
        # Add color features (9 + 9 = 18 features)
        features.extend(upper_colors.flatten())
        features.extend(lower_colors.flatten())
        
        # Add accessory features
        features.append(float(accessories.get('footwear_type', 0)) / 5)
        features.append(float(accessories.get('headwear_present', False)))
        features.append(float(accessories.get('bag_present', False)))
        features.append(float(accessories.get('bag_position', 0)) / 3)
        features.append(float(accessories.get('jacket_present', False)))
        features.append(float(accessories.get('pattern_type', 0)) / 4)
        
        # Pad to APPEARANCE_FEATURE_DIM
        features = np.array(features, dtype=np.float32)
        if len(features) < APPEARANCE_FEATURE_DIM:
            features = np.pad(features, (0, APPEARANCE_FEATURE_DIM - len(features)))
        else:
            features = features[:APPEARANCE_FEATURE_DIM]
        
        return normalize(features)
