"""
Camera Capture Module for PrivacyFlow

Thread-safe camera capture with support for:
- USB cameras (V4L2)
- Raspberry Pi Camera Module
- RTSP streams
"""

import logging
import threading
import time
from typing import Optional, Tuple, List
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration."""
    source: int | str = 0  # Camera index or RTSP URL
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 2
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0


class CameraCapture:
    """
    Thread-safe camera capture with buffering.
    
    Features:
    - Background thread for continuous capture
    - Frame buffering to handle processing delays
    - Auto-reconnect on camera failures
    - FPS monitoring
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        """
        Initialize camera capture.
        
        Args:
            config: Camera configuration. Uses defaults if None.
        """
        self.config = config or CameraConfig()
        
        self._cap = None
        self._frame_buffer: deque = deque(maxlen=self.config.buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Stats
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0.0
    
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if started successfully, False otherwise.
        """
        if self._running:
            logger.warning("Camera already running")
            return True
        
        if not self._open_camera():
            return False
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Camera started: {self.config.source}")
        return True
    
    def stop(self) -> None:
        """Stop camera capture."""
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        logger.info("Camera stopped")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest frame.
        
        Returns:
            Tuple of (success, frame). Frame is None if no frame available.
        """
        with self._lock:
            if not self._frame_buffer:
                return False, None
            return True, self._frame_buffer[-1].copy()
    
    def read_all(self) -> List[np.ndarray]:
        """
        Read all buffered frames (clears buffer).
        
        Returns:
            List of frames (oldest first).
        """
        with self._lock:
            frames = list(self._frame_buffer)
            self._frame_buffer.clear()
            return frames
    
    @property
    def fps(self) -> float:
        """Get current FPS."""
        return self._fps
    
    @property
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running
    
    def _open_camera(self) -> bool:
        """Open camera device."""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed")
            return False
        
        source = self.config.source
        
        # Handle RTSP streams
        if isinstance(source, str) and source.startswith(('rtsp://', 'http://')):
            self._cap = cv2.VideoCapture(source)
        else:
            # Try V4L2 backend on Linux for USB cameras
            try:
                self._cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            except:
                self._cap = cv2.VideoCapture(source)
        
        if not self._cap.isOpened():
            logger.error(f"Failed to open camera: {source}")
            return False
        
        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # Reduce buffer for lower latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Log actual settings
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        return True
    
    def _capture_loop(self) -> None:
        """Background capture loop."""
        consecutive_failures = 0
        max_failures = 10
        
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if self.config.auto_reconnect:
                    logger.warning(f"Camera disconnected, reconnecting in {self.config.reconnect_delay}s...")
                    time.sleep(self.config.reconnect_delay)
                    self._open_camera()
                    continue
                else:
                    break
            
            ret, frame = self._cap.read()
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive failures")
                    if self.config.auto_reconnect:
                        self._cap.release()
                        self._cap = None
                        consecutive_failures = 0
                    else:
                        break
                continue
            
            consecutive_failures = 0
            
            # Add to buffer
            with self._lock:
                self._frame_buffer.append(frame)
            
            # Update FPS
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def list_cameras(max_index: int = 10) -> List[int]:
    """
    List available camera indices.
    
    Args:
        max_index: Maximum index to check.
    
    Returns:
        List of available camera indices.
    """
    try:
        import cv2
    except ImportError:
        return []
    
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    return available


def get_camera_info(index: int = 0) -> dict:
    """
    Get information about a camera.
    
    Args:
        index: Camera index.
    
    Returns:
        Dictionary with camera properties.
    """
    try:
        import cv2
    except ImportError:
        return {"error": "OpenCV not installed"}
    
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return {"error": f"Cannot open camera {index}"}
    
    info = {
        "index": index,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "backend": cap.getBackendName(),
    }
    
    cap.release()
    return info
