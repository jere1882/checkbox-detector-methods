import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def display_detections(image: np.ndarray, 
                       filled_checkboxes: List[Tuple[int, int, int, int, float]], 
                       unfilled_checkboxes: List[Tuple[int, int, int, int, float]]) -> None:
    """
    Displays an image with detected checkboxes highlighted.
    Filled checkboxes are marked with red rectangles, and unfilled checkboxes are marked with green rectangles.

    Args:
        image (np.ndarray): Grayscale image where checkboxes are located.
        filled_checkboxes (List[Tuple[int, int, int, int]]): 
            List of bounding boxes for filled checkboxes, each as (x, y, width, height).
        unfilled_checkboxes (List[Tuple[int, int, int, int]]): 
            List of bounding boxes for unfilled checkboxes, each as (x, y, width, height).

    Returns:
        None: The function displays the image with detections using Matplotlib.
    """
    # Convert the grayscale image to RGB to enable colored drawings
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Draw red rectangles for filled checkboxes
    for (x, y, w, h, s) in filled_checkboxes:
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red box
    
    # Draw green rectangles for unfilled checkboxes
    for (x, y, w, h, s) in unfilled_checkboxes:
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Display the image with Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Checkboxes Detected")
    plt.show()

def calculate_bounding_box_intersection(bbox1, bbox2):
    """
    Calculates the intersection of two bounding boxes.

    Args:
        bbox1: First bounding box (x, y, width, height).
        bbox2: Second bounding box (x, y, width, height).

    Returns:
        intersection_area: The area of the intersection.
        bbox1_area: The area of the first bounding box.
        bbox2_area: The area of the second bounding box.
    """
    # Get the coordinates of the intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    # Calculate the width and height of the intersection
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    # Calculate the intersection area
    intersection_area = intersection_width * intersection_height

    # Calculate the area of each bounding box
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]

    return intersection_area, bbox1_area, bbox2_area


def check_bounding_box_overlap(bbox1, bbox2, threshold_percentage=0.5):
    """
    Checks if two bounding boxes overlap by a given percentage.

    Args:
        bbox1: First bounding box (x, y, width, height).
        bbox2: Second bounding box (x, y, width, height).
        threshold_percentage: Minimum overlap percentage to consider as overlapping.

    Returns:
        bool: True if the bounding boxes overlap by the specified threshold, False otherwise.
    """
    intersection_area, bbox1_area, bbox2_area = calculate_bounding_box_intersection(bbox1, bbox2)

    # Calculate the overlap ratio with respect to the area of each bounding box
    overlap_ratio1 = intersection_area / bbox1_area
    overlap_ratio2 = intersection_area / bbox2_area

    # Check if either bounding box overlaps by at least the threshold percentage
    return overlap_ratio1 >= threshold_percentage or overlap_ratio2 >= threshold_percentage