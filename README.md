# LEGO Inventory System

An automated LEGO inventory system developed using Python and Computer Vision techniques as part of a university Image Processing module.

## Project Overview

The aim of this project is to develop an image processing system capable of analysing photographs containing multiple LEGO bricks and automatically generating an inventory.

The system will process images to identify individual LEGO bricks, calculate the total number of pieces and categorise them based on colour.

### Main Objectives

- Identify and isolate individual LEGO bricks from photographs.
- Count the total number of detected bricks.
- Recognise and classify individual brick colours.
- Generate an inventory showing the quantity of each colour.
- Visually display the detected bricks and processing results.

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

## Team Responsibilities

### Danish Arifuzzaman – Image Preprocessing & Segmentation

Responsible for preparing input images and isolating LEGO bricks from their backgrounds.

Techniques:
- Grayscale and colour space conversion.
- Gaussian Blur and noise reduction.
- Image thresholding (Otsu and adaptive thresholding).
- Morphological operations (erosion, dilation, opening and closing).
- Binary mask generation.
- Watershed segmentation.
- Separation of touching objects.

**Expected Output:** Cleaned and segmented images containing individual LEGO brick regions.

### Sean Iroanya – Brick Detection & Counting

Responsible for identifying individual LEGO bricks and calculating the total number of pieces.

Techniques:
- Contour detection.
- Connected component analysis.
- Bounding boxes.
- Object area calculation.
- Noise and false detection filtering.
- Object counting.

**Expected Output:** Individually detected LEGO bricks and their total count.

### Brandon Traynor – Colour Recognition & Inventory Generation

Responsible for identifying the colour of each detected brick and generating the final inventory.

Techniques:
- HSV colour analysis.
- Colour thresholding and classification.
- Mean and dominant colour extraction.
- Colour histograms.
- Colour categorisation.
- Inventory statistics and visualisation.

**Expected Output:** A breakdown of LEGO bricks by colour and a final inventory summary.

## Overall System Workflow

```text
Input Photograph
      |
      v
Image Preprocessing & Segmentation
       (Danish)
      |
      v
Brick Detection & Counting
        (Sean)
      |
      v
Colour Recognition
       (Brandon)
      |
      v
Final LEGO Inventory
```

## Example Final Output

```text
LEGO INVENTORY
-------------------------
Total Bricks: 53

Red:       12
Blue:       8
Yellow:    15
Green:      7
Black:      6
White:      5
```