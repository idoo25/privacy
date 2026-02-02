#!/usr/bin/env python3
"""
PrivacyFlow - Main Entry Point

Privacy-preserving person detection and flow analysis system.

Usage:
    python main.py --config config.yaml --station-id station_001
    python main.py --station-id platform_3 --camera 0 --api-port 5000
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Optional

from src.privacyflow import PrivacyFlow
from src.camera import CameraCapture, list_available_cameras
from src.api import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('privacyflow.log')
    ]
)
logger = logging.getLogger('privacyflow.main')


class PrivacyFlowRunner:
    """Main runner for PrivacyFlow system."""
    
    def __init__(self, 
                 config_path: str,
                 station_id: str,
                 camera_source: int | str = 0,
                 api_port: Optional[int] = None,
                 headless: bool = True):
        """
        Initialize the runner.
        
        Args:
            config_path: Path to configuration file
            station_id: Station identifier
            camera_source: Camera index or RTSP URL
            api_port: Port for REST API (None to disable)
            headless: Run without display window
        """
        self.config_path = config_path
        self.station_id = station_id
        self.camera_source = camera_source
        self.api_port = api_port
        self.headless = headless
        
        self.privacyflow: Optional[PrivacyFlow] = None
        self.camera: Optional[CameraCapture] = None
        self._running = False
        
        # Statistics
        self.frames_processed = 0
        self.persons_detected = 0
        self.start_time: Optional[datetime] = None
    
    def setup(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if setup successful
        """
        logger.info("Setting up PrivacyFlow system...")
        
        # Initialize PrivacyFlow
        try:
            self.privacyflow = PrivacyFlow(self.config_path)
            logger.info("PrivacyFlow initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PrivacyFlow: {e}")
            return False
        
        # Initialize camera
        try:
            config = self.privacyflow.config.get('detection', {})
            self.camera = CameraCapture(
                camera_source=self.camera_source,
                width=config.get('frame_width', 640),
                height=config.get('frame_height', 480),
                fps=config.get('fps', 30)
            )
            
            if not self.camera.start():
                logger.error("Failed to open camera")
                return False
            
            logger.info(f"Camera {self.camera_source} opened successfully")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
        
        return True
    
    def run(self) -> None:
        """Run the main processing loop."""
        self._running = True
        self.start_time = datetime.now()
        
        # Start PrivacyFlow
        self.privacyflow.start()
        logger.info(f"PrivacyFlow started for station: {self.station_id}")
        
        # Optional: Start API server in background thread
        if self.api_port:
            import threading
            api_thread = threading.Thread(
                target=self._run_api_server,
                daemon=True
            )
            api_thread.start()
            logger.info(f"API server started on port {self.api_port}")
        
        # Optional: Display window
        if not self.headless:
            try:
                import cv2
                cv2.namedWindow('PrivacyFlow', cv2.WINDOW_NORMAL)
            except Exception:
                self.headless = True
                logger.warning("Cannot create display window, running headless")
        
        # Main processing loop
        logger.info("Starting main processing loop...")
        last_stats_time = time.time()
        
        try:
            while self._running:
                # Get frame from camera
                frame = self.camera.get_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                detections = self.privacyflow.process_frame(
                    frame,
                    self.station_id,
                    f"cam_{self.camera_source}"
                )
                
                self.frames_processed += 1
                self.persons_detected += detections
                
                # Display (if not headless)
                if not self.headless:
                    self._display_frame(frame, detections)
                
                # Log stats every 30 seconds
                if time.time() - last_stats_time > 30:
                    self._log_stats()
                    last_stats_time = time.time()
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
    
    def _display_frame(self, frame, detections: int) -> None:
        """Display frame with overlay (non-headless mode)."""
        try:
            import cv2
            
            # Add overlay text
            cv2.putText(
                frame,
                f"Persons: {detections}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                f"Frames: {self.frames_processed}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )
            
            cv2.imshow('PrivacyFlow', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._running = False
        except Exception:
            pass
    
    def _run_api_server(self) -> None:
        """Run API server in background."""
        try:
            app = create_app(
                self.privacyflow.database,
                self.privacyflow.trajectory_builder
            )
            app.run(host='0.0.0.0', port=self.api_port, threaded=True)
        except Exception as e:
            logger.error(f"API server error: {e}")
    
    def _log_stats(self) -> None:
        """Log current statistics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        fps = self.frames_processed / elapsed if elapsed > 0 else 0
        
        unique_tokens = self.privacyflow.trajectory_builder.count_unique_tokens()
        
        logger.info(
            f"Stats - Frames: {self.frames_processed}, "
            f"FPS: {fps:.1f}, "
            f"Persons detected: {self.persons_detected}, "
            f"Unique tokens today: {unique_tokens}"
        )
    
    def stop(self) -> None:
        """Stop all components."""
        logger.info("Stopping PrivacyFlow system...")
        
        self._running = False
        
        if self.camera:
            self.camera.stop()
            logger.info("Camera stopped")
        
        if self.privacyflow:
            self.privacyflow.stop()
            logger.info("PrivacyFlow stopped")
        
        if not self.headless:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        
        # Final stats
        self._log_stats()
        logger.info("PrivacyFlow system stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PrivacyFlow - Privacy-Preserving Person Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --station-id station_001
  python main.py --station-id platform_3 --camera 0 --api-port 5000
  python main.py --config custom_config.yaml --station-id entrance_a
  python main.py --list-cameras
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--station-id', '-s',
        required=False,
        help='Station identifier (required unless --list-cameras)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    
    parser.add_argument(
        '--rtsp',
        type=str,
        default=None,
        help='RTSP URL for network camera (overrides --camera)'
    )
    
    parser.add_argument(
        '--api-port',
        type=int,
        default=None,
        help='Port for REST API (disabled if not specified)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run without display window (default: True)'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        help='Show display window with video feed'
    )
    
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='PrivacyFlow 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Handle --list-cameras
    if args.list_cameras:
        cameras = list_available_cameras()
        if cameras:
            print("Available cameras:")
            for idx in cameras:
                print(f"  Camera {idx}")
        else:
            print("No cameras found")
        return
    
    # Validate station-id
    if not args.station_id:
        parser.error("--station-id is required")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Determine camera source
    camera_source = args.rtsp if args.rtsp else args.camera
    
    # Determine headless mode
    headless = args.headless and not args.display
    
    # Create and run
    runner = PrivacyFlowRunner(
        config_path=args.config,
        station_id=args.station_id,
        camera_source=camera_source,
        api_port=args.api_port,
        headless=headless
    )
    
    if runner.setup():
        runner.run()
    else:
        logger.error("Setup failed, exiting")
        sys.exit(1)


if __name__ == "__main__":
    main()
