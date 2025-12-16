import unittest
import cv2
from checkbox_detector import preprocess_image, detect_checkboxes, classify_checkboxes
import os
import numpy as np

class TestCheckboxDetector(unittest.TestCase):

    def setUp(self):
        # Define the paths to the test images
        self.image_filled_checkbox = os.path.join(os.path.dirname(__file__), "test_data", "filled_checkbox.jpg")
        self.image_unfilled_checkbox = os.path.join(os.path.dirname(__file__), "test_data", "unfilled_checkbox.jpg")
        self.image_filled_and_unfilled = os.path.join(os.path.dirname(__file__), "test_data", "filled_and_unfilled_checkbox.jpg")
        self.image_only_text = os.path.join(os.path.dirname(__file__), "test_data", "text_without_checkboxes.jpg")

    def test_detect_filled_checkbox(self):
        # Preprocess and detect checkboxes in the filled checkbox image
        morph = preprocess_image(self.image_filled_checkbox)
        checkboxes = detect_checkboxes(morph)    
        
        # Assert one checkbox is detected
        self.assertEqual(len(checkboxes), 1, "Should detect one checkbox in the image.")

    def test_detect_unfilled_checkbox(self):
        # Preprocess and detect checkboxes in the unfilled checkbox image
        morph = preprocess_image(self.image_unfilled_checkbox)
        checkboxes = detect_checkboxes(morph)

        # Assert one checkbox is detected
        self.assertEqual(len(checkboxes), 1, "Should detect one checkbox in the image.")
    
    def test_detect_filled_and_unfilled_checkboxes(self):
        # Preprocess and detect checkboxes in the mixed checkbox image
        morph = preprocess_image(self.image_filled_and_unfilled)
        checkboxes = detect_checkboxes(morph)

        # Assert two checkboxes are detected
        self.assertEqual(len(checkboxes), 2, "Should detect two checkboxes in the image.")
    
    def test_bounding_box_filled_checkbox(self):
        # Preprocess and detect checkboxes in the filled checkbox image
        morph = preprocess_image(self.image_filled_checkbox)
        checkboxes = detect_checkboxes(morph)
        
        # Verify the bounding box dimensions
        x, y, w, h = checkboxes[0]
        self.assertTrue(20 < w < 50, f"Expected width in range (20, 50), got {w}.")
        self.assertTrue(20 < h < 50, f"Expected height in range (20, 50), got {h}.")
    
    def test_bounding_box_unfilled_checkbox(self):
        # Preprocess and detect checkboxes in the unfilled checkbox image
        morph = preprocess_image(self.image_unfilled_checkbox)
        checkboxes = detect_checkboxes(morph)
        
        # Verify the bounding box dimensions
        x, y, w, h = checkboxes[0]
        self.assertTrue(20 < w < 50, f"Expected width in range (20, 50), got {w}.")
        self.assertTrue(20 < h < 50, f"Expected height in range (20, 50), got {h}.")
    
    def test_bounding_box_filled_and_unfilled(self):
        # Preprocess and detect checkboxes in the mixed checkbox image
        morph = preprocess_image(self.image_filled_and_unfilled)
        checkboxes = detect_checkboxes(morph)

        # Verify bounding box dimensions for both checkboxes
        for (x, y, w, h) in checkboxes:
            self.assertTrue(20 < w < 50, f"Expected width in range (20, 50), got {w}.")
            self.assertTrue(20 < h < 50, f"Expected height in range (20, 50), got {h}.")
    
    def test_classify_filled_checkbox(self):
        # Preprocess, detect checkboxes and classify in the filled checkbox image
        morph = preprocess_image(self.image_filled_checkbox)
        checkboxes = detect_checkboxes(morph)
        image = cv2.imread(self.image_filled_checkbox)
        filled_checkboxes, unfilled_checkboxes = classify_checkboxes(morph, checkboxes)

        # Assert that the detected checkbox is filled
        self.assertEqual(len(filled_checkboxes), 1, "Should detect one filled checkbox.")
        self.assertEqual(len(unfilled_checkboxes), 0, "Should detect no unfilled checkboxes.")
    
    def test_classify_unfilled_checkbox(self):
        # Preprocess, detect checkboxes and classify in the unfilled checkbox image
        morph = preprocess_image(self.image_unfilled_checkbox)
        checkboxes = detect_checkboxes(morph)
        image = cv2.imread(self.image_unfilled_checkbox)
        filled_checkboxes, unfilled_checkboxes = classify_checkboxes(morph, checkboxes)

        # Assert that the detected checkbox is unfilled
        self.assertEqual(len(filled_checkboxes), 0, "Should detect no filled checkboxes.")
        self.assertEqual(len(unfilled_checkboxes), 1, "Should detect one unfilled checkbox.")
    
    def test_classify_filled_and_unfilled_checkboxes(self):
        # Preprocess, detect checkboxes and classify in the mixed checkbox image
        morph = preprocess_image(self.image_filled_and_unfilled)
        checkboxes = detect_checkboxes(morph)
        image = cv2.imread(self.image_filled_and_unfilled)
        filled_checkboxes, unfilled_checkboxes = classify_checkboxes(morph, checkboxes)

        # Assert that one filled checkbox and one unfilled checkbox are detected
        self.assertEqual(len(filled_checkboxes), 1, "Should detect one filled checkbox.")
        self.assertEqual(len(unfilled_checkboxes), 1, "Should detect one unfilled checkbox.")
    
    def test_text_image(self):
        morph = preprocess_image(self.image_only_text)
        checkboxes = detect_checkboxes(morph)
        
        # Assert that no checkboxes are detected
        self.assertEqual(len(checkboxes), 0, "Should detect no checkboxes in the text-only image.")
    
if __name__ == "__main__":
    unittest.main()

