#!/usr/bin/env python3
"""
Batch prediction script for Gemini checkbox detection.

Generates predictions in standardized JSON format for evaluation.

Usage:
    python predict.py                           # Process all val images
    python predict.py --input ../data/val/images --output ../predictions/gemini
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from detect_checkboxes import detect_checkboxes

load_dotenv(Path(__file__).parent.parent / ".env")


def gemini_to_standard_format(detections: dict, img_w: int, img_h: int, image_name: str) -> dict:
    """Convert Gemini detections to standardized prediction format."""
    predictions = []
    
    for cb in detections.get('checkboxes', []):
        box_2d = cb.get('box_2d', [])
        if len(box_2d) != 4:
            continue
        
        # Gemini format: [y_min, x_min, y_max, x_max] normalized 0-1000
        y_min, x_min, y_max, x_max = box_2d
        
        x1 = x_min * img_w / 1000
        y1 = y_min * img_h / 1000
        x2 = x_max * img_w / 1000
        y2 = y_max * img_h / 1000
        
        class_id = 1 if cb.get('label') == 'filled_checkbox' else 0
        
        predictions.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'class_id': class_id,
            'confidence': 1.0,  # LLM doesn't provide confidence
        })
    
    return {
        'image': image_name,
        'width': img_w,
        'height': img_h,
        'predictions': predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gemini predictions in standardized format",
    )
    parser.add_argument("--input", type=str, default="../data/val/images",
                        help="Input images directory")
    parser.add_argument("--output", type=str, default="../predictions/gemini",
                        help="Output predictions directory")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro",
                        help="Gemini model to use (default: gemini-2.5-pro)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found")
        sys.exit(1)
    
    print("=" * 60)
    print("Gemini Batch Prediction")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    image_files = sorted(input_dir.glob("*.jpg"))
    print(f"Images: {len(image_files)}")
    
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {image_path.name}")
        
        try:
            # Get image dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  Error: Could not read image")
                continue
            img_h, img_w = img.shape[:2]
            
            # Run Gemini detection
            detections = detect_checkboxes(str(image_path), api_key, model=args.model)
            
            # Convert to standard format
            result = gemini_to_standard_format(detections, img_w, img_h, image_path.name)
            
            # Save
            output_path = output_dir / f"{image_path.stem}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"  ✓ {len(result['predictions'])} predictions saved")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✓ Predictions saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

