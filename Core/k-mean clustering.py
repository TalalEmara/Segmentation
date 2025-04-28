import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def initialize_centroids(data, K):
    indices = np.random.choice(data.shape[0], K, replace=False)
    return data[indices]

def compute_distances(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return distances

def update_centroids(data, labels, K):
    new_centroids = np.array([
        data[labels == k].mean(axis=0) if np.any(labels == k)
        else data[np.random.randint(0, data.shape[0])]
        for k in range(K)
    ])
    return new_centroids

def kmeans(data, K, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(data, K)
    
    for i in range(max_iters):
        old_centroids = centroids.copy()
        distances = compute_distances(data, centroids)
        labels = np.argmin(distances, axis=1)
        centroids = update_centroids(data, labels, K)

        if np.linalg.norm(centroids - old_centroids) < tol:
            print(f"Converged at iteration {i}")
            break

    return centroids, labels

def segment_image(image_array, K):
    pixels = image_array.reshape(-1, 3).astype(np.float32)

    centroids, labels = kmeans(pixels, K)

    segmented_pixels = centroids[labels].astype(np.uint8)
    segmented_image = segmented_pixels.reshape(image_array.shape)

    return segmented_image, centroids


if __name__ == "__main__":

    image_path = 'images/Butterfly.png' 
    K = 5
    image = Image.open(image_path).convert('RGB')
    
    original_image = np.array(image)
    segmented_image, centroids = segment_image(original_image, K=K)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Segmented Image (K={K})")
    plt.imshow(segmented_image)
    plt.axis('off')



    # Show cluster colors as squares
    plt.subplot(1, 3, 3)
    plt.title("Cluster Colors")

    # Create an image with K color patches
    color_patches = np.zeros((50, 50 * K, 3), dtype=np.uint8)
    for i in range(K):
        color_patches[:, i*50:(i+1)*50, :] = centroids[i].astype(np.uint8)

    plt.imshow(color_patches)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
