import argparse
from checkbox_detector import preprocess_image, detect_checkboxes, classify_checkboxes, display_detections, detect_checkboxes_simple
import cv2
import matplotlib.pyplot as plt

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Detect and classify checkboxes in an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--debug", action="store_true", default=False, help="Plot intermediate steps for debugging")
    parser.add_argument("--debug_individual_contours", action="store_true", default=False, 
                            help="Plot individual contours for debugging. Only small images.")

    args = parser.parse_args()
    
    image_path = args.image_path

    # Preprocess the image
    morph = preprocess_image(image_path)
    
    if args.debug:
        
        plt.figure(figsize=(6, 6))
        plt.imshow(morph)
        plt.title("Preprocessed image")
        plt.axis('off')
        plt.show()
    
    # Detect checkboxes
    checkboxes = detect_checkboxes(morph, args.debug, args.debug_individual_contours)

    # Load the original image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Classify checkboxes
    filled_checkboxes, unfilled_checkboxes = classify_checkboxes(morph, checkboxes)

    # Display results
    print("Num filled checkboxes:", len(filled_checkboxes))
    print("Num unfilled checkboxes:", len(unfilled_checkboxes))
    
    display_detections(image, filled_checkboxes, unfilled_checkboxes)

if __name__ == "__main__":
    main()

