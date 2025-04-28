import cv2
import numpy as np

def iterative_threshold(image, epsilon=0.5, max_iterations=200):

    height, width = image.shape

    top_left = image[0, 0]
    top_right = image[0, width - 1]
    bottom_left = image[height - 1, 0]
    bottom_right = image[height - 1, width - 1]

    initial_corners = np.array([top_left, top_right, bottom_left, bottom_right])
    current_threshold = np.mean(initial_corners)

    for iteration in range(max_iterations):

        background_pixels = image[image <= current_threshold]
        object_pixels = image[image > current_threshold]

        # Check if either group is empty
        if background_pixels.size == 0 or object_pixels.size == 0:
            print("Warning: One of the groups became empty. Stopping early.")
            break

        mean_background = np.mean(background_pixels)
        mean_object = np.mean(object_pixels)

        new_threshold = (mean_background + mean_object) / 2

        if abs(new_threshold - current_threshold) < epsilon:
            break

        current_threshold = new_threshold

    # Step 6: Apply the final threshold to create a binary image
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image > current_threshold] = 0
    binary_image[image <= current_threshold] = 255

    return current_threshold, binary_image

if __name__ == "__main__":
    # Read the image in grayscale
    img = cv2.imread('../images/Snake/fish.png', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found or path incorrect.")
        exit()

    threshold_value, binary_img = iterative_threshold(img)

    print(f"Final Threshold: {threshold_value}")

    # Show the images
    cv2.imshow('Original Image', img)
    cv2.imshow('Binary Image', binary_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
