#!/usr/bin/env python3
"""
Checkbox detection using Gemini 2.5 Pro Vision LLM.

This script sends an image to Google's Gemini API and asks it to detect
checkboxes with bounding boxes. Zero training required.

Usage:
    python detect_checkboxes.py input.jpg
    python detect_checkboxes.py input.jpg output.jpg
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Checkbox detection prompt
CHECKBOX_DETECTION_PROMPT = """Analyze this document image and detect ALL checkboxes with their bounding boxes.

DETECTION RULES:
1. Detect both FILLED (checked) and EMPTY (unchecked) checkboxes
2. A filled checkbox has a checkmark, X, or is filled in
3. An empty checkbox is an empty square or circle
4. Include ALL checkboxes, even small ones or partially visible ones

For each checkbox, provide:
- The bounding box coordinates [y_min, x_min, y_max, x_max] normalized to 0-1000
- The label: "filled_checkbox" or "empty_checkbox"

CRITICAL: Be thorough. Count every checkbox on the page. Do not skip any.
"""


def create_checkbox_detection_schema() -> dict:
    """Create the JSON schema for structured output."""
    return {
        "type": "OBJECT",
        "properties": {
            "checkboxes": {
                "type": "ARRAY",
                "description": "List of all detected checkboxes with bounding boxes",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "box_2d": {
                            "type": "ARRAY",
                            "description": "Bounding box [y_min, x_min, y_max, x_max] normalized to 0-1000",
                            "items": {"type": "INTEGER"},
                        },
                        "label": {
                            "type": "STRING",
                            "description": "Either 'filled_checkbox' or 'empty_checkbox'",
                            "enum": ["filled_checkbox", "empty_checkbox"],
                        },
                    },
                    "required": ["box_2d", "label"],
                },
            },
            "total_filled": {
                "type": "INTEGER",
                "description": "Total count of filled checkboxes",
            },
            "total_empty": {
                "type": "INTEGER",
                "description": "Total count of empty checkboxes",
            },
        },
        "required": ["checkboxes", "total_filled", "total_empty"],
    }


def load_image(image_path: str) -> tuple[bytes, str]:
    """Load image and return bytes + MIME type."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        image_bytes = f.read()

    return image_bytes, mime_type


def detect_checkboxes(image_path: str, api_key: str, model: str = "gemini-2.5-pro") -> dict:
    """
    Detect checkboxes in an image using Gemini Vision model.

    Args:
        image_path: Path to the input image
        api_key: Gemini API key
        model: Model name (e.g., "gemini-2.5-pro", "gemini-3.0-pro")

    Returns:
        Dictionary with detection results
    """
    client = genai.Client(api_key=api_key)
    image_bytes, mime_type = load_image(image_path)

    contents = [
        types.Part.from_text(text=CHECKBOX_DETECTION_PROMPT),
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
    ]

    generation_config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=create_checkbox_detection_schema(),
    )

    print(f"Sending image to {model}...")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generation_config,
    )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        print(f"Raw response: {response.text}")
        raise

    return result


def draw_detections(image_path: str, detections: dict, output_path: str) -> None:
    """
    Draw bounding boxes on the image and save.

    Args:
        image_path: Path to input image
        detections: Detection results from Gemini
        output_path: Path to save annotated image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = img.shape[:2]

    # Colors: Green for empty, Red for filled (matching YOLO style)
    colors = {
        "empty_checkbox": (0, 255, 0),    # Green (BGR)
        "filled_checkbox": (0, 0, 255),   # Red (BGR)
    }

    for checkbox in detections.get("checkboxes", []):
        box = checkbox.get("box_2d", [])
        if len(box) != 4:
            continue

        # Convert from 0-1000 normalized coords to pixel coords
        y_min, x_min, y_max, x_max = box
        x1 = int(x_min * width / 1000)
        y1 = int(y_min * height / 1000)
        x2 = int(x_max * width / 1000)
        y2 = int(y_max * height / 1000)

        label = checkbox.get("label", "empty_checkbox")
        color = colors.get(label, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(output_path, img)
    print(f"✓ Saved annotated image: {output_path}")


def run_inference(input_image_path: str, output_image_path: str = None, model: str = "gemini-2.5-pro") -> dict:
    """
    Run Gemini inference on an image.
    
    Args:
        input_image_path: Path to input image
        output_image_path: Path to save output image (auto-generated if None)
        model: Model name (e.g., "gemini-2.5-pro", "gemini-3.0-pro")
        
    Returns:
        Detection results dict
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment or .env file")

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    # Run detection
    detections = detect_checkboxes(input_image_path, api_key, model=model)

    # Create output directory
    script_dir = Path(__file__).parent
    runs_dir = script_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Generate base name from input
    input_name = Path(input_image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_path = runs_dir / f"{input_name}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"✓ Saved JSON results: {json_path}")

    # Save annotated image
    if output_image_path is None:
        output_image_path = str(runs_dir / f"{input_name}_{timestamp}_detected.jpg")
    
    draw_detections(input_image_path, detections, output_image_path)

    return detections


def main():
    parser = argparse.ArgumentParser(
        description="Detect checkboxes using Gemini 2.5 Pro Vision LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference (saves to runs/ folder)
  python detect_checkboxes.py input.jpg
  
  # Run inference with custom output path
  python detect_checkboxes.py input.jpg output.jpg
        """
    )
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("output_image", type=str, nargs="?", default=None,
                        help="Path to save output image (optional, defaults to runs/)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro",
                        help="Gemini model to use (default: gemini-2.5-pro)")

    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment or .env file")
        print("Please set it in .env file at repository root:")
        print("  GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    print("=" * 60)
    print("Gemini Checkbox Detector")
    print("=" * 60)
    print(f"Input: {args.input_image}")

    try:
        detections = run_inference(
            input_image_path=args.input_image,
            output_image_path=args.output_image,
            model=args.model,
        )
    except Exception as e:
        print(f"Detection failed: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "-" * 40)
    print("Detection Results:")
    print("-" * 40)
    print(f"  Detected {detections.get('total_filled', 0)} filled_checkbox(es)")
    print(f"  Detected {detections.get('total_empty', 0)} empty_checkbox(es)")
    print(f"  Total detections: {len(detections.get('checkboxes', []))}")

    print("\n" + "=" * 60)
    print("✓ Inference completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
