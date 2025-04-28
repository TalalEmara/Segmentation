import cv2
import numpy as np
import matplotlib.pyplot as plt

from Core.imageMode import rgb_to_grayscale

def region_growing(image, seed_point, threshold=20, use_intensity=True):
    if use_intensity:
        img = rgb_to_grayscale(image)
        seed_value = img[seed_point[1], seed_point[0]]
    else:
        img = image
        seed_value = image[seed_point[1], seed_point[0]].astype(np.int32)

    height, width = img.shape[:2]
    region = np.zeros((height, width), dtype=np.uint8)
    region[seed_point[1], seed_point[0]] = 255  # Initialize with seed

    points = [seed_point]

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while points:
        x, y = points.pop(0)

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if region[ny, nx] == 0:  # Not already added
                    if use_intensity:
                        pixel_value = img[ny, nx]
                        diff = abs(int(pixel_value) - int(seed_value))
                    else:
                        pixel_value = img[ny, nx].astype(np.int32)
                        diff = np.linalg.norm(pixel_value - seed_value)

                    if diff < threshold:
                        region[ny, nx] = 255
                        points.append((nx, ny))


    region_mask = (region == 255)

    if len(image.shape) == 3:  # Colored image
        segmented_img = np.zeros_like(image)
        for c in range(3):
            segmented_img[:, :, c][region_mask] = image[:, :, c][region_mask]
    else:  # Grayscale image
        segmented_img = np.zeros_like(image)
        segmented_img[region_mask] = image[region_mask]

    return segmented_img


img = cv2.imread('../images/Snake/apple.png')

seed = (80, 250)

segmented_region = region_growing(img, seed, threshold=90, use_intensity=True)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title('Segmented Region (Original Values)')
plt.imshow(cv2.cvtColor(segmented_region, cv2.COLOR_BGR2RGB))
plt.show()
