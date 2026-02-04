"""
Real Person Detection using YOLOv8 ONNX models

This module provides actual person detection using ONNX models
downloaded from Hugging Face (deepghs/yolo-person).
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """
    YOLOv8 person detector using ONNX Runtime.
    
    Designed to work with models from:
    - deepghs/yolo-person (person detection)
    - deepghs/yolo-face (face detection)
    """
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            input_size: Model input size (width, height)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        self.session = None
        self.input_name = None
        self.output_names = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ONNX model."""
        if not self.model_path.exists():
            logger.warning(f"Model not found: {self.model_path}")
            return
        
        try:
            import onnxruntime as ort
            
            # Use CPU provider (works everywhere)
            providers = ['CPUExecutionProvider']
            
            # Try GPU if available
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"Loaded model: {self.model_path.name}")
            logger.info(f"  Input: {self.input_name}")
            logger.info(f"  Outputs: {self.output_names}")
            logger.info(f"  Provider: {self.session.get_providers()[0]}")
            
        except ImportError:
            logger.error("onnxruntime not installed")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in frame.
        
        Args:
            frame: BGR image (H, W, C)
        
        Returns:
            List of detections, each with:
            - bbox: (x, y, w, h)
            - confidence: float
            - class_id: int
        """
        if self.session is None:
            return []
        
        # Preprocess
        input_tensor, scale, pad = self._preprocess(frame)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self._postprocess(outputs[0], frame.shape, scale, pad)
        
        return detections
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess frame for YOLOv8.
        
        Returns:
            Tuple of (input_tensor, scale, (pad_w, pad_h))
        """
        h, w = frame.shape[:2]
        target_w, target_h = self.input_size
        
        # Calculate scale (letterbox)
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        try:
            import cv2
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Fallback resize
            resized = self._simple_resize(frame, new_w, new_h)
        
        # Create canvas with padding
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Convert to tensor: (1, 3, H, W), normalized
        tensor = canvas.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
        tensor = np.expand_dims(tensor, 0)  # Add batch
        
        return tensor, scale, (pad_w, pad_h)
    
    def _simple_resize(self, img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        """Simple nearest-neighbor resize without OpenCV."""
        h, w = img.shape[:2]
        x_ratio = w / new_w
        y_ratio = h / new_h
        
        x_indices = (np.arange(new_w) * x_ratio).astype(int)
        y_indices = (np.arange(new_h) * y_ratio).astype(int)
        
        return img[y_indices][:, x_indices]
    
    def _postprocess(self, 
                     output: np.ndarray,
                     orig_shape: Tuple[int, ...],
                     scale: float,
                     pad: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Postprocess YOLOv8 output.
        
        YOLOv8 output shape: (1, 84, 8400) for COCO
        or (1, 5, N) for single-class models
        """
        # Handle different output formats
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Transpose if needed (84, 8400) -> (8400, 84)
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        # Extract boxes and scores
        # Format: [x_center, y_center, width, height, class_scores...]
        n_classes = output.shape[1] - 4
        
        boxes = output[:, :4]
        
        if n_classes == 1:
            # Single class model (person-only)
            scores = output[:, 4]
            class_ids = np.zeros(len(scores), dtype=int)
        else:
            # Multi-class model
            class_scores = output[:, 4:]
            scores = class_scores.max(axis=1)
            class_ids = class_scores.argmax(axis=1)
        
        # Filter by confidence
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert from center to corner format
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply NMS
        keep = self._nms(boxes, scores, self.iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        
        # Scale back to original image
        pad_w, pad_h = pad
        orig_h, orig_w = orig_shape[:2]
        
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            # Remove padding
            x1, y1, x2, y2 = box
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            # Convert to (x, y, w, h)
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            
            if bbox[2] > 0 and bbox[3] > 0:
                detections.append({
                    'bbox': bbox,
                    'confidence': float(score),
                    'class_id': int(class_id)
                })
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep


class MoveNetPoseEstimator:
    """
    MoveNet pose estimator using ONNX Runtime.
    
    Designed for Xenova/movenet-singlepose-lightning model.
    """
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (192, 192)):
        """
        Initialize pose estimator.
        
        Args:
            model_path: Path to ONNX model
            input_size: Model input size
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        
        self.session = None
        self.input_name = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ONNX model."""
        if not self.model_path.exists():
            logger.warning(f"Pose model not found: {self.model_path}")
            return
        
        try:
            import onnxruntime as ort
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            
            logger.info(f"Loaded pose model: {self.model_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load pose model: {e}")
    
    def estimate(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate pose keypoints.
        
        Args:
            person_crop: Cropped person image (BGR)
        
        Returns:
            17x3 array of (y, x, confidence) keypoints, or None
        """
        if self.session is None:
            return None
        
        # Resize to input size
        try:
            import cv2
            resized = cv2.resize(person_crop, self.input_size)
        except ImportError:
            return None
        
        # Prepare input (MoveNet expects int32)
        input_tensor = resized.astype(np.int32)
        input_tensor = np.expand_dims(input_tensor, 0)  # Add batch
        
        # Run inference
        try:
            outputs = self.session.run(None, {self.input_name: input_tensor})
            keypoints = outputs[0][0, 0]  # Shape: (17, 3)
            return keypoints
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return None


# Keypoint names for reference
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
