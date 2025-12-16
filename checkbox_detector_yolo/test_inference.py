#!/usr/bin/env python3
"""YOLO Checkbox Detector - Run inference on images"""

import argparse
from ultralytics import YOLO
import cv2
import os

def run_inference(input_image_path, output_image_path=None, weights_path="runs/detect/train/weights/best.pt", conf=0.2):
    """Run YOLO model inference on an image"""
    
    # Check if weights exist
    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at {weights_path}")
        return False
    
    # Check if input image exists
    if not os.path.exists(input_image_path):
        print(f"ERROR: Input image not found at {input_image_path}")
        return False
    
    print(f"Loading model from {weights_path}...")
    try:
        model = YOLO(weights_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False
    
    print(f"\nRunning inference on {input_image_path}...")
    try:
        # Run inference
        results = model.predict(input_image_path, conf=conf, verbose=False)
        
        # Extract results
        result = results[0]
        boxes = result.boxes
        
        # Count detections by class
        class_names = {0: "empty_checkbox", 1: "filled_checkbox"}
        detections = {0: 0, 1: 0}
        
        if len(boxes) > 0:
            for cls_id in boxes.cls:
                detections[int(cls_id)] += 1
        
        print(f"✓ Inference completed successfully!")
        print(f"  Detected {detections[0]} empty_checkbox(es)")
        print(f"  Detected {detections[1]} filled_checkbox(es)")
        print(f"  Total detections: {len(boxes)}")
        
        # Save output image if output path is provided
        if output_image_path:
            # Get the annotated image with predictions drawn
            annotated_image = result.plot()
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_image_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the image
            cv2.imwrite(output_image_path, annotated_image)
            print(f"✓ Output image saved to {output_image_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLO checkbox detection on an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference and save output
  python test_inference.py input.jpg output.jpg
  
  # Run inference only (no output saved)
  python test_inference.py input.jpg
  
  # Use custom model weights
  python test_inference.py input.jpg output.jpg --weights custom_model.pt
  
  # Adjust confidence threshold
  python test_inference.py input.jpg output.jpg --conf 0.3
        """
    )
    
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("output_image", type=str, nargs="?", default=None, 
                       help="Path to save output image with predictions (optional)")
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt",
                       help="Path to model weights file (default: runs/detect/train/weights/best.pt)")
    parser.add_argument("--conf", type=float, default=0.2,
                       help="Confidence threshold for detections (default: 0.2)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO Checkbox Detector")
    print("=" * 60)
    
    success = run_inference(
        input_image_path=args.input_image,
        output_image_path=args.output_image,
        weights_path=args.weights,
        conf=args.conf
    )
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Inference completed successfully")
    else:
        print("✗ Inference failed")
    print("=" * 60)

