import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.spatial.distance import cdist

class AgglomerativeClusteringSuperpixel:
    def __init__(self, n_clusters=5, n_superpixels=100):
        self.n_clusters = n_clusters
        self.n_superpixels = n_superpixels

    def _linkage(self, X, cluster1, cluster2, distances):
        # Use average linkage
        dists = [distances[i][j] for i in cluster1 for j in cluster2]
        return np.mean(dists)

    def fit_predict(self, image):
        # Convert image to float and apply SLIC superpixel segmentation
        image_float = img_as_float(image)
        segments = slic(image_float, n_segments=self.n_superpixels, compactness=10)
        
        # Calculate superpixel features (average color and position)
        superpixels = []
        features = []
        for label in np.unique(segments):
            mask = segments == label
            # Get average color
            avg_color = np.mean(image_float[mask], axis=0)
            # Get centroid position
            y, x = np.where(mask)
            centroid = np.array([np.mean(y), np.mean(x)])
            # Combine color and position (with weight for position)
            feature = np.concatenate([avg_color, centroid * 0.01])  # weight position less
            features.append(feature)
            superpixels.append(mask)
        
        features = np.array(features)
        
        # Perform agglomerative clustering on superpixel features
        clusters = [[i] for i in range(len(features))]
        distances = cdist(features, features)
        np.fill_diagonal(distances, np.inf)

        while len(clusters) > self.n_clusters:
            min_dist = np.inf
            pair_to_merge = (0, 1)

            # Find closest pair of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = self._linkage(features, clusters[i], clusters[j], distances)
                    if d < min_dist:
                        min_dist = d
                        pair_to_merge = (i, j)

            # Merge the closest pair
            i, j = pair_to_merge
            clusters[i].extend(clusters[j])
            del clusters[j]

        # Create labels for superpixels
        superpixel_labels = np.zeros(len(features), dtype=int)
        for idx, cluster in enumerate(clusters):
            for i in cluster:
                superpixel_labels[i] = idx
                
        # Map superpixel labels back to original image
        label_image = np.zeros_like(segments)
        for label, superpixel_label in enumerate(superpixel_labels):
            label_image[segments == label] = superpixel_label
            
        return label_image

def agglomerative_segment_with_superpixels(image, n_clusters=5, n_superpixels=100):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Convert BGR to RGB
    
    # Resize for faster processing (optional)
    image = cv2.resize(image, (200, 200))
    
    clustering = AgglomerativeClusteringSuperpixel(n_clusters=n_clusters, n_superpixels=n_superpixels)
    labels = clustering.fit_predict(image)
    
    # Create segmented image by averaging colors in each cluster
    segmented = np.zeros_like(image, dtype=np.float32)
    for i in range(n_clusters):
        mask = labels == i
        segmented[mask] = np.mean(image[mask], axis=0)
    
    return segmented.astype(np.uint8)

# Example usage:
if __name__ == '__main__':
    image = cv2.imread("CV/Segmentation/images/nbc.png")
    segmented_image = agglomerative_segment_with_superpixels(
        image,
        n_clusters=7,
        n_superpixels=500
    )
    cv2.imshow("Segmented Image with Superpixels", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()