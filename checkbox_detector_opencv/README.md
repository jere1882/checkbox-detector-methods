# Checkbox detection using OpenCV

This repository contains a Python package for detecting checkboxes in images. It includes preprocessing, detection, and classification modules, along with visualization utilities and unit testing. The goal is to provide an easy-to-use, modular approach for detecting checkbox-like structures in scanned documents, forms, and other images.

## Directory Structure

```bash
checkbox_detector/
├── checkbox_detector/
│   ├── __init__.py            # Initialization of the checkbox_detector package
│   ├── preprocessing.py       # Image preprocessing utilities (e.g., thresholding)
│   ├── detection.py           # Functions for detecting checkboxes in images
│   ├── classification.py      # (Future functionality) Classify detected checkboxes
│   └── utils.py               # Utility functions (e.g., bounding box intersection, overlap checking)
├── tests/
│   ├── __init__.py            # Initialization of the tests package
│   └── test_checkbox_detector.py  # Unit tests for the package
│   └── test_data              # Simple crops for unit testing
├── scripts/
│   └── run_detector.py        # Example script for detecting checkboxes using the package
├── setup.py                   # Installation script for the package
├── data/images                # Testing images
└── README.md                  # This file
```

## Installation

Clone and install the repository.

```bash
pip install .
```

## Usage

```bash
python scripts/run_detector.py data/images/d1.jpg
```

## Results

When testing on the challenge image using the default parameters, the result looks like this:

![Demo Image](demo.png)

Where green are empty checkboxes and red are filled checkboxes. 

## Issues

* Detecting boxes like the "No Zoning" one can prove quite challenging, since it is necessary to have a high level 
understanding of the image. Perpahps a learned approach would be more suitable to handle it, as demoed in the YOLO
folder.

* This implementation struggles to generalize to similar yet slightly different layouts. Testing and tuning using
a broader validation set is necessary.
