#!/usr/bin/env python3
"""
Batch prediction script for OpenCV checkbox detection.

Generates predictions in standardized JSON format for evaluation.

Usage:
    python predict.py                           # Process all val images
    python predict.py --input ../data/val/images --output ../predictions/opencv
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))
from checkbox_detector.preprocessing import preprocess_image
from checkbox_detector.detection import detect_checkboxes
from checkbox_detector.classification import classify_checkboxes


def opencv_to_standard_format(image_path: str, image_name: str) -> dict:
    """Run OpenCV detection and convert to standardized prediction format."""
    # Load image for dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_h, img_w = img.shape[:2]
    
    # Preprocess
    processed = preprocess_image(image_path)
    
    # Detect checkboxes
    checkbox_bboxes = detect_checkboxes(processed)
    
    # Classify checkboxes
    filled, unfilled = classify_checkboxes(processed, checkbox_bboxes)
    
    predictions = []
    
    # Add filled checkboxes (class_id = 1)
    for x, y, w, h, intensity in filled:
        predictions.append({
            'bbox': [float(x), float(y), float(x + w), float(y + h)],
            'class_id': 1,
            'confidence': 1.0,  # OpenCV doesn't provide confidence
        })
    
    # Add unfilled checkboxes (class_id = 0)
    for x, y, w, h, intensity in unfilled:
        predictions.append({
            'bbox': [float(x), float(y), float(x + w), float(y + h)],
            'class_id': 0,
            'confidence': 1.0,
        })
    
    return {
        'image': image_name,
        'width': img_w,
        'height': img_h,
        'predictions': predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenCV predictions in standardized format",
    )
    parser.add_argument("--input", type=str, default="../data/val/images",
                        help="Input images directory")
    parser.add_argument("--output", type=str, default="../predictions/opencv",
                        help="Output predictions directory")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("OpenCV Batch Prediction")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    image_files = sorted(input_dir.glob("*.jpg"))
    print(f"Images: {len(image_files)}")
    
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {image_path.name}")
        
        try:
            # Run detection
            result = opencv_to_standard_format(str(image_path), image_path.name)
            
            # Save
            output_path = output_dir / f"{image_path.stem}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"  ✓ {len(result['predictions'])} predictions saved")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✓ Predictions saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()



