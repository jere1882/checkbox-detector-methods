from setuptools import setup, find_packages

setup(
    name="checkbox_detector_opencv",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "opencv-python", "matplotlib"],
    description="A Python package for detecting and classifying checkboxes in images.",
    author="Jeremias Rodriguez",
    author_email="jeremiaslcc@gmail.com",
    url="https://github.com/jere1882/checkbox_detector",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

