import argparse
from checkbox_detector import preprocess_image, detect_checkboxes, classify_checkboxes
import cv2
import matplotlib.pyplot as plt
import numpy as np

def save_detections(image: np.ndarray, 
                   filled_checkboxes, 
                   unfilled_checkboxes,
                   output_path: str) -> None:
    """
    Saves an image with detected checkboxes highlighted.
    Filled checkboxes are marked with red rectangles, and unfilled checkboxes are marked with green rectangles.
    """
    # Convert the grayscale image to RGB to enable colored drawings
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Draw red rectangles for filled checkboxes
    for (x, y, w, h, s) in filled_checkboxes:
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red box
    
    # Draw green rectangles for unfilled checkboxes
    for (x, y, w, h, s) in unfilled_checkboxes:
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Save the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Checkboxes Detected")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved output to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Detect and classify checkboxes in an image and save output.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output", type=str, help="Path to save output image", default=None)

    args = parser.parse_args()
    
    image_path = args.image_path
    output_path = args.output or image_path.replace('.jpg', '_detected.png').replace('.jpeg', '_detected.png').replace('.png', '_detected.png')

    # Preprocess the image
    morph = preprocess_image(image_path)
    
    # Detect checkboxes
    checkboxes = detect_checkboxes(morph, plot_contours=False, plot_individual_contours=False, debug=False)

    # Load the original image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Classify checkboxes
    filled_checkboxes, unfilled_checkboxes = classify_checkboxes(morph, checkboxes)

    # Display results
    print("Num filled checkboxes:", len(filled_checkboxes))
    print("Num unfilled checkboxes:", len(unfilled_checkboxes))
    
    save_detections(image, filled_checkboxes, unfilled_checkboxes, output_path)

if __name__ == "__main__":
    main()

