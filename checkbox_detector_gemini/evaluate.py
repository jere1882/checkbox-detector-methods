#!/usr/bin/env python3
"""
Evaluate Gemini checkbox detection against ground truth labels.

Uses the `supervision` library for standard object detection metrics.

Usage:
    python evaluate.py                    # Evaluate all val images
    python evaluate.py --image val1.jpg   # Evaluate single image
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.metrics import F1Score, Precision, Recall, MeanAveragePrecision
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from detect_checkboxes import detect_checkboxes

load_dotenv(Path(__file__).parent.parent / ".env")

CLASS_NAMES = ['empty_checkbox', 'filled_checkbox']


def load_ground_truth(label_path: str, img_w: int, img_h: int) -> sv.Detections:
    """Load YOLO format ground truth labels as supervision Detections."""
    boxes = []
    class_ids = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                
                # Convert YOLO format to xyxy
                x1 = (x_c - w / 2) * img_w
                y1 = (y_c - h / 2) * img_h
                x2 = (x_c + w / 2) * img_w
                y2 = (y_c + h / 2) * img_h
                
                boxes.append([x1, y1, x2, y2])
                class_ids.append(class_id)
    
    if not boxes:
        return sv.Detections.empty()
    
    return sv.Detections(
        xyxy=np.array(boxes),
        class_id=np.array(class_ids),
    )


def gemini_to_detections(detections: dict, img_w: int, img_h: int) -> sv.Detections:
    """Convert Gemini detections to supervision Detections."""
    boxes = []
    class_ids = []
    confidences = []
    
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
        
        boxes.append([x1, y1, x2, y2])
        class_ids.append(class_id)
        confidences.append(1.0)  # LLM doesn't provide confidence, assume 1.0
    
    if not boxes:
        return sv.Detections.empty()
    
    return sv.Detections(
        xyxy=np.array(boxes),
        class_id=np.array(class_ids),
        confidence=np.array(confidences),
    )


def run_gemini_inference(image_path: str, api_key: str) -> sv.Detections:
    """Run Gemini detection on an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img_h, img_w = img.shape[:2]
    
    detections_raw = detect_checkboxes(image_path, api_key)
    return gemini_to_detections(detections_raw, img_w, img_h), img_w, img_h


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Gemini checkbox detection against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, help="Evaluate single image (e.g., val1.jpg)")
    parser.add_argument("--save-json", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found")
        sys.exit(1)
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "val"
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    
    # Get images to evaluate
    if args.image:
        image_files = [args.image]
    else:
        image_files = sorted([f.name for f in images_dir.glob("*.jpg")])
    
    print("=" * 60)
    print("Gemini Checkbox Detection Evaluation")
    print("=" * 60)
    print(f"Using: supervision v{sv.__version__}")
    print(f"Images: {len(image_files)}")
    print(f"Classes: {CLASS_NAMES}")
    
    # Initialize metrics
    f1_metric = F1Score()
    precision_metric = Precision()
    recall_metric = Recall()
    map_metric = MeanAveragePrecision()
    
    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    per_image_results = []
    
    for image_file in image_files:
        image_path = str(images_dir / image_file)
        label_path = str(labels_dir / image_file.replace('.jpg', '.txt'))
        
        if not os.path.exists(label_path):
            print(f"\n  Skipping {image_file} (no label file)")
            continue
        
        try:
            # Load ground truth
            img = cv2.imread(image_path)
            img_h, img_w = img.shape[:2]
            gt = load_ground_truth(label_path, img_w, img_h)
            
            # Run Gemini detection
            pred, _, _ = run_gemini_inference(image_path, api_key)
            
            print(f"\n  {image_file}: GT={len(gt)} boxes, Pred={len(pred)} boxes")
            
            # Store for batch metrics
            all_predictions.append(pred)
            all_targets.append(gt)
            
            per_image_results.append({
                'image': image_file,
                'num_gt': len(gt),
                'num_pred': len(pred),
            })
            
        except Exception as e:
            print(f"\n  Error evaluating {image_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_predictions:
        print("\nNo images evaluated!")
        return
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("Computing Metrics...")
    print("=" * 60)
    
    # Update all metrics with batch data
    f1_result = f1_metric.update(all_predictions, all_targets).compute()
    precision_result = precision_metric.update(all_predictions, all_targets).compute()
    recall_result = recall_metric.update(all_predictions, all_targets).compute()
    map_result = map_metric.update(all_predictions, all_targets).compute()
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    total_gt = sum(r['num_gt'] for r in per_image_results)
    total_pred = sum(r['num_pred'] for r in per_image_results)
    
    print(f"\nDataset Summary:")
    print(f"  Images: {len(per_image_results)}")
    print(f"  Ground Truth: {total_gt} boxes")
    print(f"  Predictions:  {total_pred} boxes")
    
    print(f"\nF1 Score:")
    print(f"  F1 @ IoU=0.50: {f1_result.f1_50:.4f}")
    print(f"  F1 @ IoU=0.75: {f1_result.f1_75:.4f}")
    
    print(f"\nPrecision:")
    print(f"  Precision @ IoU=0.50: {precision_result.precision_at_50:.4f}")
    print(f"  Precision @ IoU=0.75: {precision_result.precision_at_75:.4f}")
    
    print(f"\nRecall:")
    print(f"  Recall @ IoU=0.50: {recall_result.recall_at_50:.4f}")
    print(f"  Recall @ IoU=0.75: {recall_result.recall_at_75:.4f}")
    
    print(f"\nMean Average Precision (mAP):")
    print(f"  mAP @ IoU=0.50: {map_result.map50:.4f}")
    print(f"  mAP @ IoU=0.75: {map_result.map75:.4f}")
    print(f"  mAP @ IoU=0.50:0.95: {map_result.map50_95:.4f}")
    
    # Per-class results if available
    if hasattr(f1_result, 'f1_per_class') and f1_result.f1_per_class is not None:
        print(f"\nPer-Class F1 @ IoU=0.50:")
        for i, class_name in enumerate(CLASS_NAMES):
            if i < len(f1_result.f1_per_class):
                f1_50_class = f1_result.f1_per_class[i][0] if len(f1_result.f1_per_class[i]) > 0 else 0
                print(f"  {class_name}: {f1_50_class:.4f}")
    
    # Save JSON if requested
    if args.save_json:
        output = {
            'library': f'supervision {sv.__version__}',
            'num_images': len(per_image_results),
            'total_gt': total_gt,
            'total_pred': total_pred,
            'f1_50': float(f1_result.f1_50),
            'f1_75': float(f1_result.f1_75),
            'precision_50': float(precision_result.precision_at_50),
            'precision_75': float(precision_result.precision_at_75),
            'recall_50': float(recall_result.recall_at_50),
            'recall_75': float(recall_result.recall_at_75),
            'map50': float(map_result.map50),
            'map75': float(map_result.map75),
            'map50_95': float(map_result.map50_95),
            'per_image': per_image_results,
        }
        
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to {args.save_json}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
