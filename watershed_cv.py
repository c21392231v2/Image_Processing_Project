import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path

image_path = Path(__file__).resolve().parent / "water_coins.jpg"
img = cv.imread(str(image_path))

assert img is not None, f"Could not read image: {image_path}"

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(grey, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

#noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

#sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

#finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(unknown, cmap="gray")
plt.title("Unknown Region")
plt.axis("off")

plt.tight_layout()
plt.show()