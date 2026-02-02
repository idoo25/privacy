"""
Camera Capture Module

Real-time camera capture for Raspberry Pi and USB cameras.
"""

import threading
import time
from typing import Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class CameraCapture:
    """
    Thread-safe camera capture for real-time processing.
    
    Supports:
    - USB cameras
    - Raspberry Pi Camera Module (via V4L2)
    - RTSP streams
    """
    
    def __init__(self, 
                 camera_source: int | str = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 buffer_size: int = 1):
        """
        Initialize camera capture.
        
        Args:
            camera_source: Camera index (int) or RTSP URL (str)
            width: Frame width
            height: Frame height
            fps: Target frames per second
            buffer_size: OpenCV buffer size (1 = latest frame only)
        """
        if cv2 is None:
            raise ImportError("OpenCV required. Install with: pip install opencv-python")
        
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_frame_time = 0.0
    
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if camera opened successfully
        """
        # Open camera
        self._cap = cv2.VideoCapture(self.camera_source)
        
        if not self._cap.isOpened():
            return False
        
        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        # For Raspberry Pi camera, use V4L2
        if isinstance(self.camera_source, int):
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        # Wait for first frame
        timeout = time.time() + 5.0
        while self._frame is None and time.time() < timeout:
            time.sleep(0.1)
        
        return self._frame is not None
    
    def _capture_loop(self) -> None:
        """Continuous capture loop running in background thread."""
        while self._running:
            ret, frame = self._cap.read()
            
            if ret:
                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
                    self._last_frame_time = time.time()
            else:
                # Camera disconnected - attempt reconnect
                time.sleep(1.0)
                self._cap.release()
                self._cap = cv2.VideoCapture(self.camera_source)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame.
        
        Returns:
            Frame as numpy array (BGR format) or None if no frame available
        """
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None
    
    def get_frame_if_new(self, last_count: int) -> Tuple[Optional[np.ndarray], int]:
        """
        Get frame only if newer than last_count.
        
        Args:
            last_count: Previous frame count
        
        Returns:
            Tuple of (frame or None, current frame count)
        """
        with self._frame_lock:
            if self._frame_count > last_count:
                return self._frame.copy(), self._frame_count
            return None, self._frame_count
    
    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count
    
    @property
    def actual_fps(self) -> float:
        """Get actual FPS based on frame timing."""
        if self._last_frame_time == 0:
            return 0.0
        elapsed = time.time() - self._last_frame_time
        if elapsed > 1.0:
            return 0.0  # No recent frames
        return self._frame_count / max(1, time.time() - self._last_frame_time)
    
    @property
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running and self._cap is not None and self._cap.isOpened()
    
    def stop(self) -> None:
        """Stop camera capture and release resources."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._cap:
            self._cap.release()
            self._cap = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class MultiCameraCapture:
    """Manage multiple cameras simultaneously."""
    
    def __init__(self):
        self.cameras: dict[str, CameraCapture] = {}
    
    def add_camera(self, camera_id: str, camera: CameraCapture) -> None:
        """Add a camera to the manager."""
        self.cameras[camera_id] = camera
    
    def start_all(self) -> dict[str, bool]:
        """Start all cameras. Returns dict of camera_id -> success."""
        results = {}
        for camera_id, camera in self.cameras.items():
            results[camera_id] = camera.start()
        return results
    
    def get_frames(self) -> dict[str, Optional[np.ndarray]]:
        """Get frames from all cameras."""
        return {
            camera_id: camera.get_frame()
            for camera_id, camera in self.cameras.items()
        }
    
    def stop_all(self) -> None:
        """Stop all cameras."""
        for camera in self.cameras.values():
            camera.stop()


# Raspberry Pi specific utilities
def get_pi_camera_info() -> dict:
    """Get Raspberry Pi camera information."""
    info = {
        "available": False,
        "model": None,
        "resolution": None
    }
    
    try:
        # Check if camera module is connected
        import subprocess
        result = subprocess.run(
            ["vcgencmd", "get_camera"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "detected=1" in result.stdout:
            info["available"] = True
            info["model"] = "Raspberry Pi Camera Module"
    except Exception:
        pass
    
    return info


def list_available_cameras(max_index: int = 10) -> list[int]:
    """List available camera indices."""
    if cv2 is None:
        return []
    
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
