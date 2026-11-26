from pathlib import Path
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Choose the image and processing methods from the terminal.
parser = argparse.ArgumentParser(description="Count objects in an image.")

parser.add_argument(
    "image",
    type=Path,
    help="Path to the input image"
)

parser.add_argument(
    "--foreground",
    choices=["auto", "red"],
    default="auto",
    help="Use transparency or compare colours with the image border, or select red objects"
)

parser.add_argument(
    "--method",
    choices=["contours", "watershed"],
    default="contours",
    help="Use contours for separated objects or watershed for touching objects"
)

parser.add_argument(
    "--seed-fraction",
    type=float,
    default=0.70,
    help="Distance threshold for watershed markers, between 0 and 1"
)

args = parser.parse_args()

if not 0 < args.seed_fraction < 1:
    parser.error("--seed-fraction must be between 0 and 1")


# Load the image without discarding its transparency channel.
raw = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)

if raw is None:
    raise FileNotFoundError(f"Cannot read: {args.image.resolve()}")

# This prototype expects standard 8-bit images.
if raw.dtype != np.uint8:
    raise ValueError("Please use an 8-bit image.")

if raw.ndim == 2:
    image = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
elif raw.ndim == 3 and raw.shape[2] in (3, 4):
    image = raw[:, :, :3].copy()
else:
    raise ValueError(f"Unsupported image shape: {raw.shape}")


# Create a foreground mask: white means object, black means background.
has_transparency = (
    raw.ndim == 3
    and raw.shape[2] == 4
    and (raw[:, :, 3] < 255).any()
)

if args.foreground == "red":
    # Red appears at both ends of OpenCV's HSV hue scale.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = cv2.inRange(
        hsv, (0, 55, 30), (10, 255, 255)
    )
    upper_red = cv2.inRange(
        hsv, (170, 55, 30), (179, 255, 255)
    )

    mask = cv2.bitwise_or(lower_red, upper_red)

    # If transparency exists, exclude invisible pixels too.
    if has_transparency:
        mask[raw[:, :, 3] <= 127] = 0

    foreground_description = "Red colour mask"

elif has_transparency:
    _, mask = cv2.threshold(
        raw[:, :, 3], 127, 255, cv2.THRESH_BINARY
    )
    foreground_description = "Transparency mask"

else:
    # Estimate the background colour from the outer edge of the photo.
    # This works best when the edges are mostly clear of objects.
    softened = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(softened, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]))
    background = np.median(border, axis=0)

    # Measure brightness and colour separately so pale coloured bricks
    # can still stand out even when their brightness matches the table.
    border_difference = border - background
    difference = lab - background
    border_lightness = np.abs(border_difference[:, 0])
    border_chroma = np.linalg.norm(border_difference[:, 1:], axis=1)
    lightness_limit = max(24.0, np.percentile(border_lightness, 95) + 8)
    chroma_limit = max(12.0, np.percentile(border_chroma, 95) + 5)

    foreground = (
        (np.abs(difference[:, :, 0]) > lightness_limit)
        | (np.linalg.norm(difference[:, :, 1:], axis=2) > chroma_limit)
    )
    mask = foreground.astype(np.uint8) * 255

    # Close small gaps in the detected regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    foreground_description = "Lab border-colour mask"


# Ignore very small detected objects.
minimum_area = image.shape[0] * image.shape[1] * 0.001
objects = []
markers = None

if args.method == "contours":
    # Each separate foreground region is an object candidate.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    objects = [
        contour for contour in contours
        if cv2.contourArea(contour) >= minimum_area
    ]

else:
    # Find interior regions to use as watershed starting markers.
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    seeds = (
        distance > args.seed_fraction * distance.max()
    ).astype(np.uint8)

    # Remove very small marker regions.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seeds, connectivity=8
    )

    clean_seeds = np.zeros_like(seeds)

    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            clean_seeds[labels == label] = 1

    marker_count, markers = cv2.connectedComponents(clean_seeds)
    print(f"Starting markers: {marker_count - 1}")

    if marker_count < 2:
        raise ValueError(
            "No suitable watershed markers found. "
            "Try a lower --seed-fraction or inspect the foreground mask."
        )

    # Marker 1 is background; markers 2 onward identify objects.
    markers += 1

    # Foreground pixels outside the markers are initially unknown.
    markers[(mask > 0) & (clean_seeds == 0)] = 0

    # Grow the regions using colour changes in the original image.
    markers = cv2.watershed(image.copy(), markers)

    # Extract an object contour from each resulting region.
    for label in range(2, marker_count + 1):
        region = np.zeros_like(mask)
        region[(markers == label) & (mask > 0)] = 255

        contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(contour) >= minimum_area:
                objects.append(contour)


# Number objects from left to right.
objects.sort(key=lambda contour: cv2.boundingRect(contour)[0])

# Display transparent areas on black instead of showing hidden colours.
display_image = image.copy()

if has_transparency:
    display_image[raw[:, :, 3] <= 127] = 0

result = display_image.copy()

if markers is not None:
    # Yellow lines show watershed boundaries inside the foreground.
    result[(markers == -1) & (mask > 0)] = (0, 255, 255)

for number, contour in enumerate(objects, start=1):
    x, y, width, height = cv2.boundingRect(contour)

    cv2.rectangle(
        result,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        str(number),
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

print(f"Image: {args.image.name}")
print(f"Foreground: {foreground_description}")
print(f"Counting method: {args.method}")
print(f"Objects counted: {len(objects)}")


# Display the input, foreground mask and numbered result.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
axes[1].set_title(foreground_description)

axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Count: {len(objects)}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
