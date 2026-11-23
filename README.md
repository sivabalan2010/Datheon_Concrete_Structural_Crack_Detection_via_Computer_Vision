
# Concrete Structural Crack Detection via Computer Vision

An image-processing application for identifying potential crack patterns in concrete surfaces using computer vision and deep learning.

## Project Overview

This project provides an interactive system to process concrete surface images, enhance image quality, detect potential crack edges, and generate inspection reports.

## Features

- Upload concrete images.
- Enhance image contrast using CLAHE.
- Apply Gaussian Blur for noise reduction.
- Detect edges using Canny Edge Detection.
- Calculate detected edge pixel percentage.
- Display original and processed images.
- Generate downloadable PDF inspection reports.
- ResNet50-based crack classification model.
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

Concrete Crack Images for Classification

Source: Mendeley Data

https://data.mendeley.com/datasets/5y9wdsg2zt/2

The dataset contains positive (cracked) and negative (uncracked) concrete images.

## Installation

Install dependencies:

pip install -r requirements.txt

## Run Application

streamlit run app.py

## Workflow

Image Upload → Preprocessing → Contrast Enhancement → Edge Detection → Results → PDF Report

## Project Structure

- app.py - Image processing application and PDF report generation
- report.py - PDF report generator
- train.py - ResNet50 classification training script
- unet.py - U-Net segmentation architecture
- src/ - Image processing modules
- requirements.txt - Project dependencies

## Current Implementation Status

- Task 1: Image preprocessing and Canny edge detection implemented.
- Task 2: ResNet50 training script prepared; training and evaluation pending.
- Task 3: U-Net architecture implemented; segmentation training pending.
- Task 4: PDF report generation integrated into the Streamlit application; testing pending.

## Disclaimer

This is an educational prototype. Edge detection results do not confirm structural cracks or replace professional structural inspection.
