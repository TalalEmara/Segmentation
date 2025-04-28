import cv2 as cv
import numpy as np
from sklearn.neighbors import KDTree
import random
from imageMode import rgb_to_yuv

def mean_shift(image, bandwidth=2.0, threshold=1e-3, return_clustered_image=True, original_shape=None):
    data_points = image.reshape(-1, 3)

    data_points = np.array(data_points)
    n_points, n_features = data_points.shape

    # Initialize
    unvisited = set(range(n_points))
    cluster_centers = []
    labels = np.full(n_points, -1)  # -1 means unassigned

    # Build KDTree for fast neighbor search
    tree = KDTree(data_points)

    while unvisited:
        idx = random.sample(sorted(unvisited), 1)[0]
        center = data_points[idx]

        while True:
            # Find neighbors within bandwidth
            indices = tree.query_radius(center.reshape(1, -1), r=bandwidth)[0]
            neighbors = data_points[indices]

            # Compute new mean
            new_center = np.mean(neighbors, axis=0)

            # Check convergence
            shift_distance = np.linalg.norm(new_center - center)
            if shift_distance < threshold:
                # Try merging with existing clusters
                merged = False
                for c_idx, c in enumerate(cluster_centers):
                    if np.linalg.norm(new_center - c) < 0.5 * bandwidth:
                        cluster_centers[c_idx] = (c + new_center) / 2
                        labels[indices] = c_idx
                        merged = True
                        break

                if not merged:
                    cluster_centers.append(new_center)
                    new_label = len(cluster_centers) - 1
                    labels[indices] = new_label

                unvisited.difference_update(indices)
                break

            else:
                center = new_center  # Continue shifting

    cluster_centers = np.array(cluster_centers)

    if return_clustered_image and original_shape is not None:
        # Create clustered version of the image
        clustered_data = cluster_centers[labels]
        clustered_image = clustered_data.reshape(original_shape)
        return cluster_centers, labels, clustered_image

    return cluster_centers, labels


def mean_shift_cv(image, spatial_radius=10, color_radius=20):
    return cv.pyrMeanShiftFiltering(image, sp=spatial_radius, sr=color_radius)

def test():
    image = cv.imread('CV/Segmentation/images/Butterfly.png')
    image = cv.resize(image, (400, 400))  # (width, height)
    clustered_image = mean_shift(image, bandwidth=30, return_clustered_image=True, original_shape=(400, 400, 3))
    shiftedCV = mean_shift_cv(image, spatial_radius=10, color_radius=20)

    combined = np.hstack((clustered_image, shiftedCV))
    cv.imshow('Original vs Mean Shift', combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

test()