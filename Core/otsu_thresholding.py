import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_histogram(image):
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    hist = np.zeros(256)
    for pixel in image.ravel():
        hist[pixel] += 1
    hist /= image.size  
    return hist

import numpy as np
from PIL import Image

def otsu_threshold(image):
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    hist = np.zeros(256)
    for pixel in image.ravel():
        hist[pixel] += 1
    hist /= image.size
    
    best_threshold = 0
    max_variance = 0
    total_mean = np.sum(np.arange(256) * hist)
    
    q1 = 0.0  # Cumulative probability for class 1
    mu1 = 0.0  # Mean for class 1
    mu2 = 0.0  # Mean for class 2
    
    for t in range(256):
        # Class 1 (pixels <= t)
        q1 += hist[t]
        if q1 == 0:
            continue
        
        # Class 2 (pixels > t)
        q2 = 1.0 - q1
        if q2 == 0:
            break
        
        # Update means
        mu1 = (mu1 * (q1 - hist[t]) + t * hist[t]) / q1
        mu2 = (total_mean - q1 * mu1) / q2
        
        # Calculate between-class variance
        between_variance = q1 * q2 * (mu1 - mu2) ** 2
        
        # Update best threshold if variance is higher
        if between_variance > max_variance:
            max_variance = between_variance
            best_threshold = t
    
    thresholded_img = np.zeros_like(image)
    thresholded_img[image > best_threshold] = 255
    return best_threshold, thresholded_img

def plot_histogram(image, threshold=None):
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    hist = calculate_histogram(image) * image.size  
    plt.bar(np.arange(256), hist)
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--')
    plt.title('Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load grayscale image
    image = Image.open('images/bobama.jpg').convert('L')
    
   
   
    threshold, binary_image = otsu_threshold(image)
    print("Optimal Otsu Threshold:", threshold)
    
    # Display images
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # Histogram
    plt.subplot(1, 3, 2)
    plt.title("Histogram")
    plt.hist(np.array(image).flatten(), bins=256, range=(0, 256), color='gray')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    plt.legend()
    
    # Binarized image
    plt.subplot(1, 3, 3)
    plt.title("Binarized Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()