import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from .utils import check_bounding_box_overlap

def detect_checkboxes(
    processed_image: np.ndarray, 
    plot_contours: bool = False, 
    plot_individual_contours: bool = False,
    debug = False
) -> List[Tuple[int, int, int, int]]:
    """
    Detects checkbox-like contours in a preprocessed binary image.
    
    Args:
        processed_image (np.ndarray): A binary image where contours are to be detected.
        plot_contours (bool): If True, plots all detected contours on the image.
        plot_individual_contours (bool): If True, plots each individual contour for debugging.
        
    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding rectangles around detected checkboxes.
                                         Each rectangle is represented as (x, y, width, height).
    """
    # Find contours in the processed image.
    # - contours is a large tuple, each element a ndarray representing a contour
    # - RETR_TREE: Retrieve full hierarchy
    # - CHAIN_APPROX_SIMPLE: Simplifies lines as start and endpoint.
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if plot_contours:
        # Plot all detected contours
        image_copy = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for visualization
        cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)  # Draw contours in green

        plt.figure(figsize=(6, 6))
        plt.imshow(image_copy)
        plt.title("Contours Detected")
        plt.axis('off')
        plt.show()
        
    checkboxes = []
    areas = []  # Store the areas of detected checkboxes for filtering outliers

    for idx, contour in enumerate(contours):

        # Approximate the contour's polygon
        # - arcLength calculates the polygon of the contour
        # - 0.02 * arcLength -> the precision. A small coeffitient means more complex polygons.
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    
        if plot_individual_contours:
            image_copy = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(image_copy, contours, idx, (0, 0, 255), 2)  # Red contour for the current iteration
            
            plt.figure(figsize=(6, 6))
            plt.imshow(image_copy)
            plt.title(f"Contour {idx + 1} with len {len(approx)}")
            plt.axis('off')
            plt.show()

        # Compute bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x,y,w,h)

        # Check whether it's a nearly duplicate of the last confirmed checkbox
        if checkboxes and check_bounding_box_overlap(checkboxes[-1], bbox, 0.9):
            if debug:
                print(f"contour {idx+1} is considered a duplicate")
            continue

        # Check if the contour is square-ish
        aspect_ratio = float(w) / h
        if 0.8 < aspect_ratio < 1.2:  # Aspect ratio close to 1
            # Check convex hull alignment with bounding box
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            rect_area = w * h
            
            # If the convex hull area is close to the bounding box area, classify as a rectangle
            if np.abs(hull_area - rect_area) < 0.15 * rect_area:  # Adjust tolerance as needed
                checkboxes.append(bbox)
                areas.append(rect_area)
            elif debug:
                diff = np.abs(hull_area - rect_area) / rect_area
                print(f"contour {idx+1} is not square based on convell hull {diff}")
        elif debug:
            print(f"contour {idx+1} is not square based on aspect ratio")


    # Filter outliers based on area if multiple checkboxes are detected
    if len(areas) > 1:
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        lower_bound = mean_area - 2 * std_area
        upper_bound = mean_area + 2 * std_area
        
        # Remove outlier rectangles
        checkboxes = [
            (x, y, w, h) for (x, y, w, h), area in zip(checkboxes, areas) 
            if lower_bound <= area <= upper_bound
        ]


    return checkboxes

def detect_checkboxes_simple(processed_image: np.ndarray,
    plot_contours: bool = False, 
    plot_individual_contours: bool = False,
    debug = False) -> List[Tuple[int, int, int, int]]:
    """
    Detects simple rectangular checkbox-like contours in a preprocessed binary image.
    
    Args:
        processed_image (np.ndarray): A binary image where contours are to be detected.
        
    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding rectangles around detected checkboxes.
                                         Each rectangle is represented as (x, y, width, height).
    """
    # Find contours in the processed image
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    checkboxes = []
    for contour in contours:
        # Approximate the contour with a simple polygon to reduce the number of points
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        # Check if the contour is a rectangle (4 points and convex)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            
            # Filter by size and aspect ratio (to identify checkbox candidates)
            if 10 < w < 100 and abs(h - w) <= 0.2 * w:  # Square-ish shapes
                checkboxes.append((x, y, w, h))

    return checkboxes


# TODO: Try out cv.matchTemplate(image, templ, method)