import cv2
import numpy as np
from typing import List, Tuple

def classify_checkboxes(image: np.ndarray, checkboxes: List[Tuple[int, int, int, int]]) -> Tuple[
    List[Tuple[int, int, int, int, float]], List[Tuple[int, int, int, int, float]]]:
    """
    Classifies checkboxes in an image as filled or unfilled based on their mean intensity.

    Args:
        image (np.ndarray): Preprocessed grayscale image where checkboxes are located.
        checkboxes (List[Tuple[int, int, int, int]]): List of bounding boxes representing the checkboxes.
            Each bounding box is a tuple (x, y, width, height).
    
    Returns:
        Tuple[List[Tuple[int, int, int, int, float]], List[Tuple[int, int, int, int, float]]]: 
            A tuple containing two lists:
                - The first list contains bounding boxes of filled checkboxes.
                - The second list contains bounding boxes of unfilled checkboxes.
    """
    filled_checkboxes = []
    unfilled_checkboxes = []

    for (x, y, w, h) in checkboxes:
        # Extract the region of interest (ROI) for the checkbox

        offset = int(w * 0.05)

        roi = image[y+offset:y + h - offset, x+offset:x + w-offset]
        
        # Calculate the mean intensity of the ROI
        mean_intensity = cv2.mean(roi)[0]
        
        # Classify based on intensity: high intensity implies a filled checkbox
        if mean_intensity < 85:  # Threshold for filled checkboxes
            unfilled_checkboxes.append((x, y, w, h, mean_intensity))
        else:
            filled_checkboxes.append((x, y, w, h, mean_intensity))

    return filled_checkboxes, unfilled_checkboxes
