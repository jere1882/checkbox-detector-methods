import cv2
import numpy as np

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocesses an image by converting it to grayscale and applying adaptive thresholding.
    
    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        np.ndarray: A binary image (black and white) obtained after adaptive thresholding.
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Apply adaptive thresholding to handle varying lighting conditions
    adaptive_thresh = cv2.adaptiveThreshold(
        image, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, # We invert the values here!! white becomes 0
        blockSize=11, 
        C=2
    )

    '''
    The goal of thresholding is to convert the image into a binary mask.

    Adaptive threshold calculates, for each pixel:

    threshold = gaussian-weighted-sum(neighbourhood) - C

    if pixel > threshold => maxvalue
    else  => 0

    since it's BINARY_INV ; we invert the grayscale meanings. 0 means white, 255 means black.

    simplified version:

     cv2.threshold(img, maxval = 255, thresh=80, type=cv2.THRESH_BINARY_INV)
    '''
    
    return adaptive_thresh
