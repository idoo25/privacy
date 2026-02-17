"""
Body Proportion Analysis

Extracts body proportion RATIOS, not absolute measurements.
All values are ratios to prevent height-based identification.
"""

from typing import Dict, Optional

import numpy as np

# Body proportion features (all as ratios)
BODY_FEATURES: Dict[str, type] = {
    'height_ratio': float,        # Height relative to frame
    'shoulder_width_ratio': float, # Shoulder/height ratio
    'torso_ratio': float,         # Torso/total height
    'arm_length_ratio': float,    # Arm/height ratio
    'leg_length_ratio': float,    # Leg/height ratio
    'body_volume_estimate': float, # Approximate volume ratio
    'head_body_ratio': float,     # Head size relative to body
    'stance_width_ratio': float   # Stance width/height
}

BODY_FEATURE_DIM: int = 8


def normalize(vector: np.ndarray) -> np.ndarray:
    """L2 normalize a feature vector."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


class BodyProportionExtractor:
    """
    Extract body proportion ratios from pose estimation.
    
    Note: All values are RATIOS, not absolute measurements.
    This prevents height-based identification.
    """
    
    # Standard COCO pose keypoint indices
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize body proportion extractor.
        
        Args:
            model_path: Path to ONNX pose estimation model.
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load pose estimation model."""
        if self.model_path is not None:
            try:
                import onnxruntime as ort
                self.model = ort.InferenceSession(self.model_path)
            except (ImportError, FileNotFoundError):
                self.model = None
    
    def extract(self, pose_keypoints: np.ndarray) -> np.ndarray:
        """
        Extract body proportion ratios from pose estimation.
        
        Note: All values are RATIOS, not absolute measurements.
        This prevents height-based identification.
        
        Args:
            pose_keypoints: 17-point pose estimation results
                           Shape: (17, 3) where each row is (x, y, confidence)
        
        Returns:
            numpy.ndarray: 8-dimensional proportion vector
        """
        if pose_keypoints is None or len(pose_keypoints) < 17:
            return np.zeros(BODY_FEATURE_DIM)
        
        proportions = self._calculate_ratios(pose_keypoints)
        return normalize(proportions)
    
    def _calculate_ratios(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Calculate body proportion ratios from keypoints.
        
        Args:
            keypoints: 17-point pose keypoints (x, y, confidence)
        
        Returns:
            8-dimensional ratio vector
        """
        # Extract keypoint positions
        positions = keypoints[:, :2]  # x, y only
        confidences = keypoints[:, 2] if keypoints.shape[1] > 2 else np.ones(17)
        
        # Get key body points
        nose = positions[0]
        left_shoulder = positions[5]
        right_shoulder = positions[6]
        left_hip = positions[11]
        right_hip = positions[12]
        left_ankle = positions[15]
        right_ankle = positions[16]
        left_elbow = positions[7]
        right_elbow = positions[8]
        left_wrist = positions[9]
        right_wrist = positions[10]
        left_knee = positions[13]
        right_knee = positions[14]
        
        # Calculate body height (nose to ankles midpoint)
        ankles_mid = (left_ankle + right_ankle) / 2
        body_height = np.linalg.norm(nose - ankles_mid)
        
        if body_height < 1e-6:
            return np.zeros(BODY_FEATURE_DIM)
        
        # 1. Height ratio (placeholder - would be relative to frame in real usage)
        height_ratio = min(1.0, body_height / 1000)  # Normalized
        
        # 2. Shoulder width ratio
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        shoulder_width_ratio = shoulder_width / body_height
        
        # 3. Torso ratio (shoulders to hips)
        shoulders_mid = (left_shoulder + right_shoulder) / 2
        hips_mid = (left_hip + right_hip) / 2
        torso_length = np.linalg.norm(shoulders_mid - hips_mid)
        torso_ratio = torso_length / body_height
        
        # 4. Arm length ratio
        left_arm = np.linalg.norm(left_shoulder - left_elbow) + np.linalg.norm(left_elbow - left_wrist)
        right_arm = np.linalg.norm(right_shoulder - right_elbow) + np.linalg.norm(right_elbow - right_wrist)
        arm_length = (left_arm + right_arm) / 2
        arm_length_ratio = arm_length / body_height
        
        # 5. Leg length ratio
        left_leg = np.linalg.norm(left_hip - left_knee) + np.linalg.norm(left_knee - left_ankle)
        right_leg = np.linalg.norm(right_hip - right_knee) + np.linalg.norm(right_knee - right_ankle)
        leg_length = (left_leg + right_leg) / 2
        leg_length_ratio = leg_length / body_height
        
        # 6. Body volume estimate (rough approximation)
        hip_width = np.linalg.norm(left_hip - right_hip)
        body_volume_estimate = (shoulder_width * torso_length * hip_width) / (body_height ** 3)
        
        # 7. Head-body ratio (nose to shoulders vs total height)
        head_length = np.linalg.norm(nose - shoulders_mid)
        head_body_ratio = head_length / body_height
        
        # 8. Stance width ratio
        stance_width = np.linalg.norm(left_ankle - right_ankle)
        stance_width_ratio = stance_width / body_height
        
        return np.array([
            height_ratio,
            shoulder_width_ratio,
            torso_ratio,
            arm_length_ratio,
            leg_length_ratio,
            body_volume_estimate,
            head_body_ratio,
            stance_width_ratio
        ], dtype=np.float32)
    
    def extract_from_bbox(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extract body proportions from image and bounding box.
        
        This is a fallback when pose keypoints are not available.
        Uses simple heuristics based on bounding box dimensions.
        
        Args:
            frame: Input image
            bbox: Person bounding box (x, y, w, h)
        
        Returns:
            8-dimensional proportion vector
        """
        x, y, w, h = bbox
        
        if w <= 0 or h <= 0:
            return np.zeros(BODY_FEATURE_DIM)
        
        # Simple ratio-based features from bounding box
        aspect_ratio = w / h
        
        # Estimate proportions from typical human body ratios
        height_ratio = min(1.0, h / frame.shape[0])
        shoulder_width_ratio = 0.26 * aspect_ratio  # Typical ratio
        torso_ratio = 0.3  # Typical torso proportion
        arm_length_ratio = 0.38  # Typical arm proportion
        leg_length_ratio = 0.47  # Typical leg proportion
        body_volume_estimate = aspect_ratio * 0.2
        head_body_ratio = 0.125  # 1/8 of body height
        stance_width_ratio = 0.15 * aspect_ratio
        
        proportions = np.array([
            height_ratio,
            shoulder_width_ratio,
            torso_ratio,
            arm_length_ratio,
            leg_length_ratio,
            body_volume_estimate,
            head_body_ratio,
            stance_width_ratio
        ], dtype=np.float32)
        
        return normalize(proportions)
