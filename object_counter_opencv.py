#Here’s a complete, runnable Python program using OpenCV to perform connected components analysis and count objects in a binary image.
#It works for both grayscale and color images, and includes robust input validation and edge case handling.

#Code: Object Counting with Connected Components


import cv2
import numpy as np
import sys

def count_objects(image_path, min_area=10):
    """
    Counts connected components (objects) in a binary image using OpenCV.
    
    Parameters:
        image_path (str): Path to the input image.
        min_area (int): Minimum pixel area to consider as an object.
    
    Returns:
        int: Number of detected objects.
    """
    # Validate file path
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found or unsupported format: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Convert to binary using Otsu's threshold
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Filter out background (label 0) and small objects
    object_count = 0
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            object_count += 1

    print(f"Detected objects: {object_count}")

    # Optional: visualize results
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = centroids[i]
            cv2.circle(output, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    cv2.imshow("Detected Objects", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return object_count

if __name__ == "__main__":
    # Example usage
    image_path = "objects.png"  # Replace with your image file
    count_objects(image_path, min_area=20)


#How It Works:


#Image Loading & Validation
#Ensures the file exists and is readable.


#Binary Conversion
#Uses Otsu’s thresholding to automatically binarize the image.


#Connected Components Analysis
#cv2.connectedComponentsWithStats returns:

#num_labels: total labels (including background)
#labels: label map for each pixel
#stats: bounding box and area for each component
#centroids: center coordinates



#Filtering Small Objects
#Removes noise by ignoring components smaller than min_area.


#Visualization
#Draws bounding boxes and centroids for detected objects.



#Example Run:
#If objects.png contains 5 distinct blobs:
#Detected objects: 5

#And a window will show each object outlined in green with its centroid marked in red.

#If you want, I can also give you a version that works directly on live webcam feed so it counts objects in real-time.
#Do you want me to prepare that?
