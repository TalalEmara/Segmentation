import numpy as np
import cv2 as cv
import time
from sklearn.neighbors import KDTree

def mean_shift(image, bandwidth=2.0, threshold=1e-3, bin_size=1.0):
    # Image preprocessing
    original_shape = image.shape
    image = cv.GaussianBlur(image, (5, 5), 0)
    image = cv.cvtColor(image, cv.COLOR_RGB2LUV)
    flat_image = image.reshape(-1, 3)

    data_points = np.array(flat_image)
    n_points, n_features = data_points.shape

    # Binning: quantize points to reduce redundant seed points
    quantized_points = (data_points / bin_size).astype(int)
    _, unique_indices = np.unique(quantized_points, axis=0, return_index=True)
    seed_indices = sorted(set(unique_indices))  # use one point per bin as a seed

    # KDTree for neighbor search
    tree = KDTree(data_points)

    cluster_centers = []
    labels = np.full(n_points, -1)  # -1 means unassigned

    for idx in seed_indices:
        point = data_points[idx]

        # Early convergence check: skip if already near a known mode
        is_near_mode = False
        for c in cluster_centers:
            if np.linalg.norm(point - c) < 0.5 * bandwidth:
                is_near_mode = True
                break
        if is_near_mode:
            continue

        center = point.copy()
        for _ in range(100):  # max iterations
            indices = tree.query_radius(center.reshape(1, -1), r=bandwidth)[0]
            if len(indices) == 0:
                break
            neighbors = data_points[indices]
            new_center = np.mean(neighbors, axis=0)

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
                break
            center = new_center

    cluster_centers = np.array(cluster_centers)
    clustered_data = cluster_centers[labels]
    clustered_image = clustered_data.reshape(original_shape)
    clustered_image = np.clip(clustered_image, 0, 255).astype(np.uint8)
    clustered_image = cv.cvtColor(clustered_image, cv.COLOR_LUV2RGB)
    return clustered_image


def mean_shift_cv(image, spatial_radius=10, color_radius=20):
    return cv.pyrMeanShiftFiltering(image, sp=spatial_radius, sr=color_radius)

def test():
    image = cv.imread('CV/Segmentation/images/Butterfly.png')

    # Time your custom mean shift
    start_time = time.time()
    clustered_image = mean_shift(image, bandwidth=30)
    end_time = time.time()
    print(f"Custom Mean Shift Time: {end_time - start_time:.2f} seconds")

    # Time OpenCV's mean shift
    start_time = time.time()
    shiftedCV = mean_shift_cv(image, spatial_radius=10, color_radius=20)
    end_time = time.time()
    print(f"OpenCV Mean Shift Time: {end_time - start_time:.2f} seconds")

    # Show side-by-side result
    combined = np.hstack((clustered_image, shiftedCV))
    cv.imshow('Custom Mean Shift vs OpenCV Mean Shift', combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

test()