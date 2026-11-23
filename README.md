# Datheon_Concrete_Structural_Crack_Detection_via_Computer_Vision
Concrete structural crack detection using computer vision and deep learning
# Concrete Structural Crack Detection via Computer Vision

## About
An image-processing application for identifying potential crack patterns in concrete surfaces using OpenCV.

## Objectives
- Load concrete surface images.
- Enhance image contrast using CLAHE.
- Apply Gaussian Blur.
- Detect edges using the Canny algorithm.
- Display processed results.

## Technologies
- Python
- OpenCV
- NumPy
- Streamlit

## Installation
pip install -r requirements.txt

## Run
streamlit run app.py

## Workflow
Image Upload → Preprocessing → Edge Detection → Results

## Disclaimer
This is an educational prototype and does not replace professional structural inspection.

## PDF Report Generation

The project includes a PDF report generator using ReportLab.

It records:
- Project information
- Uploaded image filename
- Detected edge pixel percentage
- Processing method
- Inspection disclaimer

## Current Implementation
- Task 1: OpenCV preprocessing and Canny edge detection
- Task 4: Basic PDF report generation
