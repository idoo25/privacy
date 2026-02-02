#!/usr/bin/env python3
"""
Download Models for PrivacyFlow

Downloads pre-trained ONNX models from Hugging Face for:
- Person detection (YOLOv8n)
- Face detection (YOLOv8n)  
- Pose estimation (MoveNet Lightning)

Usage:
    python scripts/download_models_hf.py
    python scripts/download_models_hf.py --all
    python scripts/download_models_hf.py --model person_detector
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, HfApi
except ImportError:
    print("Error: huggingface_hub not installed.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


# Model configurations
MODELS = {
    "person_detector": {
        "description": "YOLOv8n person detection model",
        "repo_id": "deepghs/yolo-person",
        "filename": "yolov8n-person/model.onnx",
        "output_name": "person_detector.onnx",
        "size_mb": 12,
        "license": "model-distribution-disclaimer-license"
    },
    "face_detector": {
        "description": "YOLOv8n face detection model",
        "repo_id": "deepghs/yolo-face",
        "filename": "yolov8n-face/model.onnx",
        "output_name": "face_detector.onnx",
        "size_mb": 12,
        "license": "model-distribution-disclaimer-license"
    },
    "pose_estimator": {
        "description": "MoveNet Lightning pose estimation model",
        "repo_id": "Xenova/movenet-singlepose-lightning",
        "filename": "onnx/model.onnx",
        "output_name": "pose_estimator.onnx",
        "size_mb": 9,
        "license": "apache-2.0"
    },
    "pose_estimator_multi": {
        "description": "MoveNet Multipose Lightning (multiple people)",
        "repo_id": "Xenova/movenet-multipose-lightning",
        "filename": "onnx/model.onnx",
        "output_name": "pose_estimator_multi.onnx",
        "size_mb": 19,
        "license": "apache-2.0",
        "optional": True
    },
    "face_encoder": {
        "description": "ArcFace face embedding model (optional - large)",
        "repo_id": "garavv/arcface-onnx",
        "filename": "arcface.onnx",
        "output_name": "face_encoder.onnx",
        "size_mb": 250,
        "license": "apache-2.0",
        "optional": True,
        "note": "Large model - consider using SimplePrivacyFaceEncoder instead for privacy"
    }
}

# Required models for basic operation
REQUIRED_MODELS = ["person_detector", "pose_estimator"]

# Recommended models
RECOMMENDED_MODELS = ["person_detector", "face_detector", "pose_estimator"]


def get_model_dir():
    """Get the models directory path."""
    script_dir = Path(__file__).parent.parent
    return script_dir / "models"


def download_model(model_name: str, model_dir: Path, force: bool = False) -> bool:
    """
    Download a single model from Hugging Face.
    
    Args:
        model_name: Name of the model to download
        model_dir: Directory to save the model
        force: Force re-download even if file exists
    
    Returns:
        True if successful, False otherwise
    """
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    config = MODELS[model_name]
    output_path = model_dir / config["output_name"]
    
    # Check if already exists
    if output_path.exists() and not force:
        print(f"  [SKIP] {model_name} already exists at {output_path}")
        return True
    
    print(f"  Downloading {model_name}...")
    print(f"    Repository: {config['repo_id']}")
    print(f"    File: {config['filename']}")
    print(f"    Size: ~{config['size_mb']}MB")
    
    if config.get("note"):
        print(f"    Note: {config['note']}")
    
    try:
        # Download from Hugging Face
        downloaded_path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
            cache_dir=model_dir / ".cache"
        )
        
        # Copy to output location
        shutil.copy(downloaded_path, output_path)
        
        # Verify file
        if output_path.exists():
            actual_size = output_path.stat().st_size / (1024 * 1024)
            print(f"    ✓ Downloaded to {output_path} ({actual_size:.1f}MB)")
            return True
        else:
            print(f"    ✗ Failed to save model")
            return False
            
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return False


def list_models():
    """Print available models."""
    print("\n📦 Available Models for PrivacyFlow:\n")
    
    print("Required for basic operation:")
    for name in REQUIRED_MODELS:
        config = MODELS[name]
        print(f"  • {name}: {config['description']} (~{config['size_mb']}MB)")
    
    print("\nRecommended:")
    for name in RECOMMENDED_MODELS:
        if name not in REQUIRED_MODELS:
            config = MODELS[name]
            print(f"  • {name}: {config['description']} (~{config['size_mb']}MB)")
    
    print("\nOptional:")
    for name, config in MODELS.items():
        if config.get("optional"):
            print(f"  • {name}: {config['description']} (~{config['size_mb']}MB)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download ONNX models for PrivacyFlow from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models_hf.py                  # Download recommended models
  python download_models_hf.py --all            # Download all models
  python download_models_hf.py --model pose_estimator  # Download specific model
  python download_models_hf.py --list           # List available models
  python download_models_hf.py --minimal        # Download only required models
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Download specific model by name"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all available models (including optional)"
    )
    
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Download only required models"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if models exist"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to save models (default: ./models)"
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list:
        list_models()
        return
    
    # Determine model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = get_model_dir()
    
    # Create directory
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Model directory: {model_dir.absolute()}\n")
    
    # Determine which models to download
    if args.model:
        models_to_download = [args.model]
    elif args.all:
        models_to_download = list(MODELS.keys())
    elif args.minimal:
        models_to_download = REQUIRED_MODELS
    else:
        models_to_download = RECOMMENDED_MODELS
    
    # Download models
    print(f"📥 Downloading {len(models_to_download)} model(s)...\n")
    
    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name, model_dir, args.force):
            success_count += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"✅ Downloaded {success_count}/{len(models_to_download)} models successfully")
    
    if success_count < len(models_to_download):
        print(f"⚠️  {len(models_to_download) - success_count} model(s) failed to download")
    
    # Show model files
    print(f"\n📁 Models in {model_dir}:")
    for f in sorted(model_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  • {f.name} ({size_mb:.1f}MB)")
    
    print("\n🚀 Ready to use with PrivacyFlow!")
    print("   Run: python main.py --station-id station_001\n")


if __name__ == "__main__":
    main()
