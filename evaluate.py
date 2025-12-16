#!/usr/bin/env python3
"""
Method-agnostic evaluator for checkbox detection.

Evaluates predictions from any method (Gemini, YOLO, OpenCV) against ground truth.
Predictions must be in standardized JSON format.

PREDICTION FORMAT (per image JSON file):
{
    "image": "val1.jpg",
    "width": 768,
    "height": 1024,
    "predictions": [
        {"bbox": [x1, y1, x2, y2], "class_id": 0, "confidence": 0.95},
        {"bbox": [x1, y1, x2, y2], "class_id": 1, "confidence": 0.87},
        ...
    ]
}

Where:
    - bbox: pixel coordinates [x_min, y_min, x_max, y_max]
    - class_id: 0 = empty_checkbox, 1 = filled_checkbox
    - confidence: detection confidence (0-1), use 1.0 if not available

Usage:
    python evaluate.py predictions/gemini                    # Evaluate single method
    python evaluate.py predictions/gemini predictions/yolo   # Compare methods
    python evaluate.py predictions/*                         # All methods
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

CLASS_NAMES = ['empty_checkbox', 'filled_checkbox']
COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for empty, Red for filled


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


def load_predictions(pred_path: str) -> tuple[sv.Detections, int, int]:
    """Load predictions from standardized JSON format."""
    with open(pred_path, 'r') as f:
        data = json.load(f)
    
    img_w = data['width']
    img_h = data['height']
    
    boxes = []
    class_ids = []
    confidences = []
    
    for pred in data.get('predictions', []):
        bbox = pred['bbox']
        boxes.append(bbox)
        class_ids.append(pred['class_id'])
        confidences.append(pred.get('confidence', 1.0))
    
    if not boxes:
        return sv.Detections.empty(), img_w, img_h
    
    return sv.Detections(
        xyxy=np.array(boxes),
        class_id=np.array(class_ids),
        confidence=np.array(confidences),
    ), img_w, img_h


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter)


def compute_metrics_at_iou(predictions: list, targets: list, iou_threshold: float) -> dict:
    """Compute precision, recall, F1 at a custom IoU threshold."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, gt in zip(predictions, targets):
        if len(pred) == 0 and len(gt) == 0:
            continue
        
        if len(pred) == 0:
            total_fn += len(gt)
            continue
        
        if len(gt) == 0:
            total_fp += len(pred)
            continue
        
        # Match predictions to ground truth
        matched_gt = set()
        tp = 0
        
        for i in range(len(pred)):
            best_iou = 0
            best_gt_idx = -1
            
            for j in range(len(gt)):
                if j in matched_gt:
                    continue
                iou = compute_iou(pred.xyxy[i], gt.xyxy[j])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
        
        total_tp += tp
        total_fp += len(pred) - tp
        total_fn += len(gt) - len(matched_gt)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_method(pred_dir: Path, data_dir: Path) -> dict:
    """Evaluate all predictions in a directory against ground truth."""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    
    # Initialize metrics
    f1_metric = F1Score()
    precision_metric = Precision()
    recall_metric = Recall()
    map_metric = MeanAveragePrecision()
    
    all_predictions = []
    all_targets = []
    per_image = []
    
    pred_files = sorted(pred_dir.glob("*.json"))
    
    for pred_file in pred_files:
        image_name = pred_file.stem + ".jpg"
        label_path = labels_dir / (pred_file.stem + ".txt")
        
        if not label_path.exists():
            continue
        
        pred, img_w, img_h = load_predictions(str(pred_file))
        gt = load_ground_truth(str(label_path), img_w, img_h)
        
        all_predictions.append(pred)
        all_targets.append(gt)
        per_image.append({
            'image': image_name,
            'num_gt': len(gt),
            'num_pred': len(pred),
        })
    
    if not all_predictions:
        return {'error': 'No predictions found'}
    
    # Compute standard metrics
    f1_result = f1_metric.update(all_predictions, all_targets).compute()
    precision_result = precision_metric.update(all_predictions, all_targets).compute()
    recall_result = recall_metric.update(all_predictions, all_targets).compute()
    map_result = map_metric.update(all_predictions, all_targets).compute()
    
    # Compute lenient metrics (IoU = 0.25, 0.30, and 0.35)
    metrics_25 = compute_metrics_at_iou(all_predictions, all_targets, 0.25)
    metrics_30 = compute_metrics_at_iou(all_predictions, all_targets, 0.30)
    metrics_35 = compute_metrics_at_iou(all_predictions, all_targets, 0.35)
    
    return {
        'method': pred_dir.name,
        'num_images': len(per_image),
        'total_gt': sum(p['num_gt'] for p in per_image),
        'total_pred': sum(p['num_pred'] for p in per_image),
        # Lenient metrics
        'f1_25': metrics_25['f1'],
        'f1_35': metrics_35['f1'],
        'precision_30': metrics_30['precision'],
        'recall_30': metrics_30['recall'],
        # Standard metrics
        'f1_50': float(f1_result.f1_50),
        'f1_75': float(f1_result.f1_75),
        'precision_50': float(precision_result.precision_at_50),
        'precision_75': float(precision_result.precision_at_75),
        'recall_50': float(recall_result.recall_at_50),
        'recall_75': float(recall_result.recall_at_75),
        # mAP metrics (VOC-style and COCO)
        'map50': float(map_result.map50),  # mAP@0.50 (VOC-style)
        'map75': float(map_result.map75),  # AP@0.75
        'map50_95': float(map_result.map50_95),  # mAP@0.50-0.95 (COCO AP)
        'per_image': per_image,
    }


def draw_predictions(image_path: str, pred_path: str, output_path: str, method_name: str):
    """Draw predictions on an image."""
    img = cv2.imread(image_path)
    if img is None:
        return
    
    pred, _, _ = load_predictions(pred_path)
    
    for i in range(len(pred)):
        x1, y1, x2, y2 = pred.xyxy[i].astype(int)
        class_id = pred.class_id[i]
        color = COLORS[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    cv2.imwrite(output_path, img)


def generate_visualizations(pred_dirs: list[Path], data_dir: Path, output_dir: Path):
    """Generate visualization images for all methods."""
    images_dir = data_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pred_dir in pred_dirs:
        method_name = pred_dir.name
        method_output = output_dir / method_name
        method_output.mkdir(exist_ok=True)
        
        for pred_file in pred_dir.glob("*.json"):
            image_name = pred_file.stem + ".jpg"
            image_path = images_dir / image_name
            
            if image_path.exists():
                output_path = method_output / image_name
                draw_predictions(str(image_path), str(pred_file), str(output_path), method_name)


def print_comparison_table(results: list[dict]):
    """Print a comparison table of all methods."""
    print("\n" + "=" * 120)
    print("COMPARISON TABLE")
    print("=" * 120)
    
    # Header - showing all requested metrics
    print(f"\n{'Method':<10} {'GT':>5} {'Pred':>5} {'R@0.3':>7} {'P@0.3':>7} {'mAP@50':>8} {'AP@75':>7} {'mAP@50-95':>11}")
    print("-" * 120)
    
    for r in results:
        if 'error' in r:
            print(f"{r['method']:<10} ERROR: {r['error']}")
        else:
            print(f"{r['method']:<10} {r['total_gt']:>5} {r['total_pred']:>5} "
                  f"{r['recall_30']:>7.3f} {r['precision_30']:>7.3f} "
                  f"{r['map50']:>8.3f} {r['map75']:>7.3f} {r['map50_95']:>11.3f}")
    
    print("-" * 120)
    
    # Additional detailed metrics section
    print("\n" + "=" * 120)
    print("DETAILED METRICS")
    print("=" * 120)
    print(f"\n{'Method':<10} {'F1@25':>7} {'F1@35':>7} {'F1@50':>7} {'F1@75':>7} "
          f"{'P@50':>7} {'R@50':>7} {'P@75':>7} {'R@75':>7}")
    print("-" * 120)
    
    for r in results:
        if 'error' not in r:
            print(f"{r['method']:<10} "
                  f"{r['f1_25']:>7.3f} {r['f1_35']:>7.3f} {r['f1_50']:>7.3f} {r['f1_75']:>7.3f} "
                  f"{r['precision_50']:>7.3f} {r['recall_50']:>7.3f} "
                  f"{r['precision_75']:>7.3f} {r['recall_75']:>7.3f}")
    
    print("-" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkbox detection methods against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("prediction_dirs", nargs='+', type=str,
                        help="Directories containing prediction JSON files")
    parser.add_argument("--data-dir", type=str, default="data/val",
                        help="Directory containing ground truth (default: data/val)")
    parser.add_argument("--output-dir", type=str, default="evaluation_output",
                        help="Directory to save visualizations (default: evaluation_output)")
    parser.add_argument("--save-json", type=str,
                        help="Save detailed results to JSON file")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization images")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    pred_dirs = [Path(p) for p in args.prediction_dirs]
    
    print("=" * 80)
    print("Checkbox Detection Evaluation")
    print("=" * 80)
    print(f"Using: supervision v{sv.__version__}")
    print(f"Ground truth: {data_dir}")
    print(f"Methods: {[p.name for p in pred_dirs]}")
    
    # Evaluate each method
    all_results = []
    for pred_dir in pred_dirs:
        if not pred_dir.exists():
            print(f"\nWarning: {pred_dir} not found, skipping")
            continue
        
        print(f"\nEvaluating: {pred_dir.name}")
        results = evaluate_method(pred_dir, data_dir)
        all_results.append(results)
        
        if 'error' not in results:
            print(f"  Images: {results['num_images']}")
            print(f"  Recall@0.3: {results['recall_30']:.4f}")
            print(f"  Precision@0.3: {results['precision_30']:.4f}")
            print(f"  mAP@0.50 (VOC): {results['map50']:.4f}")
            print(f"  AP@0.75: {results['map75']:.4f}")
            print(f"  mAP@0.50-0.95 (COCO): {results['map50_95']:.4f}")
    
    # Print comparison
    print_comparison_table(all_results)
    
    # Generate visualizations
    if args.visualize:
        print(f"\nGenerating visualizations in {output_dir}...")
        generate_visualizations(pred_dirs, data_dir, output_dir)
        print("✓ Visualizations saved")
    
    # Save JSON
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {args.save_json}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

