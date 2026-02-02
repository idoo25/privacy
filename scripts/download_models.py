"""
Download lightweight models for PrivacyFlow.

This script downloads pre-trained ONNX models for:
- Face encoding (32x32 input, 64-dim output)
- Pose estimation (lightweight)
- Appearance encoding

Usage:
    python scripts/download_models.py --lightweight
"""

import argparse
import os
from pathlib import Path


def download_models(lightweight: bool = True, output_dir: str = "models"):
    """
    Download models for PrivacyFlow.
    
    Args:
        lightweight: If True, download lightweight models suitable for edge devices
        output_dir: Directory to save models
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Model download directory: {output_path.absolute()}")
    print()
    
    if lightweight:
        print("Lightweight mode selected - using minimal models for edge devices.")
        print()
    
    # Model specifications
    models = {
        "face_encoder_32x32.onnx": {
            "description": "Face encoder (32x32 input, 64-dim output)",
            "url": None,  # Placeholder - actual URL would go here
            "size_mb": 2.5
        },
        "pose_lite.onnx": {
            "description": "Lightweight pose estimation (17 keypoints)",
            "url": None,
            "size_mb": 8.0
        },
        "appearance_encoder.onnx": {
            "description": "Appearance feature encoder (32-dim output)",
            "url": None,
            "size_mb": 5.0
        },
        "person_detector_nano.onnx": {
            "description": "YOLO-Nano person detector",
            "url": None,
            "size_mb": 4.0
        }
    }
    
    print("Required models:")
    print("-" * 60)
    for name, info in models.items():
        status = "[ ]"  # Not downloaded
        model_path = output_path / name
        if model_path.exists():
            status = "[x]"  # Downloaded
        print(f"  {status} {name}")
        print(f"      {info['description']}")
        print(f"      Size: ~{info['size_mb']} MB")
        print()
    
    print("-" * 60)
    print()
    print("NOTE: Model URLs are placeholders.")
    print("For a production system, you would need to:")
    print("  1. Train or obtain appropriate models")
    print("  2. Convert them to ONNX format")
    print("  3. Host them for download")
    print()
    print("For testing without models, PrivacyFlow uses fallback")
    print("feature extraction methods that don't require neural networks.")


def main():
    parser = argparse.ArgumentParser(
        description="Download models for PrivacyFlow"
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Download lightweight models for edge devices"
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save models (default: models)"
    )
    
    args = parser.parse_args()
    
    download_models(
        lightweight=args.lightweight,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
