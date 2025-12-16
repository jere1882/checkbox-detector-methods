#!/usr/bin/env python3
"""
Batch prediction script for YOLO checkbox detection.

Generates predictions in standardized JSON format for evaluation.

Usage:
    python predict.py                           # Process all val images
    python predict.py --input ../data/val/images --output ../predictions/yolo
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def yolo_to_standard_format(result, image_name: str) -> dict:
    """Convert YOLO results to standardized prediction format."""
    boxes = result.boxes
    img_h, img_w = result.orig_shape
    
    predictions = []
    
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        class_id = int(boxes.cls[i].cpu().numpy())
        confidence = float(boxes.conf[i].cpu().numpy())
        
        predictions.append({
            'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
            'class_id': class_id,
            'confidence': confidence,
        })
    
    return {
        'image': image_name,
        'width': img_w,
        'height': img_h,
        'predictions': predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO predictions in standardized format",
    )
    parser.add_argument("--input", type=str, default="../data/val/images",
                        help="Input images directory")
    parser.add_argument("--output", type=str, default="../predictions/yolo",
                        help="Output predictions directory")
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt",
                        help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.2,
                        help="Confidence threshold")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found: {args.weights}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("YOLO Batch Prediction")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Weights: {args.weights}")
    print(f"Confidence: {args.conf}")
    
    # Load model once
    print("\nLoading model...")
    model = YOLO(args.weights)
    print("✓ Model loaded")
    
    image_files = sorted(input_dir.glob("*.jpg"))
    print(f"Images: {len(image_files)}")
    
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {image_path.name}")
        
        try:
            # Run inference
            results = model.predict(str(image_path), conf=args.conf, verbose=False)
            result = results[0]
            
            # Convert to standard format
            output = yolo_to_standard_format(result, image_path.name)
            
            # Save
            output_path = output_dir / f"{image_path.stem}.json"
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"  ✓ {len(output['predictions'])} predictions saved")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✓ Predictions saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


