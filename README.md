# Concrete Structural Crack Detection via Computer Vision

An image-processing application for identifying potential crack patterns in concrete surfaces using computer vision and deep learning.

## Project Overview

This project provides an interactive system to process concrete surface images, enhance image quality, detect potential crack edges, and generate downloadable inspection reports.

## Features

- Upload concrete images.
- Enhance image contrast using CLAHE.
- Apply Gaussian Blur for noise reduction.
- Detect edges using Canny Edge Detection.
- Calculate detected edge pixel percentage.
- Display original and processed images.
- Generate downloadable PDF inspection reports.
- ResNet50-based crack classification training script.
- U-Net architecture for image segmentation.

## Technologies

- Python
- OpenCV
- NumPy
- Streamlit
- PyTorch
- Torchvision
- Pillow
- ReportLab

## Dataset

**Concrete Crack Images for Classification**

Source: Mendeley Data

https://data.mendeley.com/datasets/5y9wdsg2zt/2

The dataset contains positive (cracked) and negative (uncracked) concrete images.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Workflow

Image Upload → Preprocessing → Contrast Enhancement → Edge Detection → Results → PDF Report

## Project Structure

```text
Datheon_Concrete_Structural_Crack_Detection_via_Computer_Vision/
│
├── app.py
├── report.py
├── train.py
├── unet.py
├── requirements.txt
├── README.md
│
└── src/
    └── __init__.py
```

## Current Implementation Status

- **Task 1:** Image preprocessing, contrast enhancement, and Canny edge detection implemented and tested.
- **Task 2:** ResNet50 classification training script prepared; model training and evaluation pending.
- **Task 3:** U-Net segmentation architecture implemented; segmentation training and evaluation pending.
- **Task 4:** PDF inspection report generation integrated into the Streamlit application and tested successfully.

## Disclaimer

This is an educational prototype. Edge detection identifies image boundaries and does not confirm structural cracks. The generated report is not a professional structural safety assessment.
