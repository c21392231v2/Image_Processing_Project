from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Load the photograph.
image_path = Path("multiple_lego_pieces.png")  # Replace with your image file
raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

if raw is None:
    raise FileNotFoundError(f"Cannot read: {image_path.resolve()}")

if raw.ndim != 3 or raw.shape[2] != 4:
    raise ValueError(f"Expected a 4-channel image, got shape: {raw.shape}")

image = raw[:, :, :3].copy()
alpha = raw[:, :, 3]

# Use transparency to identify the foreground.
_, foreground = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

#Remove hidden background colours from the image.
image[foreground == 0] = 0

# Measure each foreground pixel's distance from the background.
distance = cv2.distanceTransform(
    foreground, cv2.DIST_L2, 5
)

# Select interior regions to separate narrow contacts between balls.
_, mask = cv2.threshold(
    distance,
    0.5 * distance.max(),
    255,
    cv2.THRESH_BINARY
)

mask = mask.astype("uint8")

# Find the outer boundary of each white region
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

minimum_area = image.shape[0] * image.shape[1] * 0.001
objects = [
    contour for contour in contours
    if cv2.contourArea(contour) >= minimum_area
]

# Number the objects from left to right.
objects.sort(key=lambda contour: cv2.boundingRect(contour)[0])

result = image.copy()

for number, contour in enumerate(objects, start=1):
    x, y, width, height = cv2.boundingRect(contour)

    cv2.rectangle(
        result, (x, y), (x + width, y + height),
        (0, 255, 0), 2
    )

    cv2.putText(
        result, str(number), (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )

print(f"Objects counted: {len(objects)}")


# Display the original, segmentation mask and numbered result.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Object interior regions")

axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Count: {len(objects)}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()