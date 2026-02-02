"""
PrivacyFlow Command Line Interface

Usage:
    privacyflow run --station-id STATION_ID [--camera INDEX] [--config PATH]
    privacyflow test-privacy
    privacyflow download-models
    privacyflow --version
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_run(args):
    """Run the PrivacyFlow detection system."""
    import numpy as np
    
    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False
        logger.warning("OpenCV not available - using simulated frames")
    
    from privacyflow.core import PrivacyFlow
    
    # Load configuration
    config_path = args.config if args.config else None
    
    logger.info(f"Starting PrivacyFlow...")
    logger.info(f"  Station ID: {args.station_id}")
    logger.info(f"  Camera: {args.camera}")
    logger.info(f"  Config: {config_path or 'default'}")
    
    # Initialize system
    pf = PrivacyFlow(config_path=config_path)
    pf.start()
    
    # Initialize camera
    cap = None
    if HAS_CV2 and args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {args.camera}")
            cap = None
    
    frame_count = 0
    detection_count = 0
    
    try:
        logger.info("Processing frames... (Press Ctrl+C to stop)")
        
        while pf.is_running:
            # Get frame
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
            else:
                # Simulated frame for testing without camera
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Process frame
            events = pf.process_frame(
                frame,
                station_id=args.station_id,
                camera_id=f"camera_{args.camera or 0}"
            )
            
            frame_count += 1
            detection_count += len(events)
            
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames, {detection_count} detections")
            
            # Display if requested and available
            if args.display and HAS_CV2:
                # Draw detection info
                info_text = f"Detections: {len(events)} | Total: {detection_count}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('PrivacyFlow', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Simulated delay if no real camera
            if cap is None:
                import time
                time.sleep(0.1)
                
                # Stop after a few frames in simulation mode
                if frame_count >= 10:
                    logger.info("Simulation complete (10 frames)")
                    break
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
    
    finally:
        pf.stop()
        if cap is not None:
            cap.release()
        if HAS_CV2:
            cv2.destroyAllWindows()
    
    # Print summary
    logger.info(f"Summary: {frame_count} frames, {detection_count} detections")
    
    # Export daily summary
    summary = pf.export_daily_summary()
    logger.info(f"Daily summary: {summary}")


def cmd_test_privacy(args):
    """Run privacy tests to verify system guarantees."""
    import subprocess
    
    logger.info("Running privacy tests...")
    
    # Try to run pytest
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_privacy.py", "-v"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            sys.exit(1)
    except FileNotFoundError:
        logger.error("pytest not installed. Install with: pip install pytest")
        sys.exit(1)


def cmd_download_models(args):
    """Download required ML models."""
    logger.info("Downloading models...")
    
    try:
        from huggingface_hub import hf_hub_download
        HAS_HF = True
    except ImportError:
        HAS_HF = False
        logger.error("huggingface_hub not installed.")
        logger.error("Install with: pip install huggingface_hub")
        sys.exit(1)
    
    models_dir = Path(args.output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    MODELS = {
        "person_detector.onnx": ("deepghs/yolo-person", "yolov8n-person/model.onnx"),
        "face_detector.onnx": ("deepghs/yolo-face", "yolov8n-face/model.onnx"),
        "pose_estimator.onnx": ("Xenova/movenet-singlepose-lightning", "onnx/model.onnx"),
    }
    
    for output_name, (repo_id, filename) in MODELS.items():
        output_path = models_dir / output_name
        
        if output_path.exists() and not args.force:
            logger.info(f"  [SKIP] {output_name} already exists")
            continue
        
        logger.info(f"  Downloading {output_name} from {repo_id}...")
        try:
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
            import shutil
            shutil.copy(downloaded, output_path)
            logger.info(f"  [OK] Saved to {output_path}")
        except Exception as e:
            logger.error(f"  [FAIL] {e}")
    
    logger.info(f"Models saved to {models_dir.absolute()}")


def cmd_info(args):
    """Show system information."""
    from privacyflow import __version__
    from privacyflow.features.face import FACE_RESOLUTION, FACE_EMBEDDING_DIM
    from privacyflow.features.body import BODY_FEATURE_DIM
    from privacyflow.features.appearance import APPEARANCE_FEATURE_DIM
    
    print(f"""
PrivacyFlow v{__version__}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Privacy Settings:
  Face Resolution:      {FACE_RESOLUTION[0]}x{FACE_RESOLUTION[1]} pixels (intentionally low)
  Face Embedding:       {FACE_EMBEDDING_DIM} dimensions
  Body Features:        {BODY_FEATURE_DIM} dimensions  
  Appearance Features:  {APPEARANCE_FEATURE_DIM} dimensions
  Total Feature Vector: {FACE_EMBEDDING_DIM + BODY_FEATURE_DIM + APPEARANCE_FEATURE_DIM} dimensions

Privacy Guarantees:
  ✓ Face resolution too low for biometric identification
  ✓ Daily salt rotation prevents cross-day tracking
  ✓ Automatic data purge at midnight
  ✓ No raw features stored - only hashed tokens
  ✓ All processing happens locally

GDPR Compliance:
  ✓ Data Minimization
  ✓ Purpose Limitation  
  ✓ Storage Limitation
  ✓ No Biometric Templates
""")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PrivacyFlow - Privacy-Preserving Person Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the detection system")
    run_parser.add_argument("--station-id", "-s", required=True, help="Station identifier")
    run_parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    run_parser.add_argument("--config", help="Path to config file")
    run_parser.add_argument("--display", "-d", action="store_true", help="Display video window")
    
    # Test command
    test_parser = subparsers.add_parser("test-privacy", help="Run privacy tests")
    
    # Download command
    download_parser = subparsers.add_parser("download-models", help="Download ML models")
    download_parser.add_argument("--output-dir", "-o", default="models", help="Output directory")
    download_parser.add_argument("--force", "-f", action="store_true", help="Force re-download")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    args = parser.parse_args()
    
    if args.version:
        from privacyflow import __version__
        print(f"PrivacyFlow v{__version__}")
        return
    
    if args.command == "run":
        cmd_run(args)
    elif args.command == "test-privacy":
        cmd_test_privacy(args)
    elif args.command == "download-models":
        cmd_download_models(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
