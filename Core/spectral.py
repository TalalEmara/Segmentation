import numpy as np
import cv2
from skimage.filters import threshold_multiotsu
from itertools import combinations
import time

def compute_histogram(image, levels=64):
    hist, _ = np.histogram(image.ravel(), bins=levels, range=(0, 256))
    return hist.astype(np.float64) / image.size  # normalize

def between_class_variance(hist, thresholds, total_mean):
    thresholds = (0,) + thresholds + (len(hist),)
    var = 0.0
    for i in range(len(thresholds) - 1):
        start, end = thresholds[i], thresholds[i + 1]
        prob = np.sum(hist[start:end])
        if prob == 0:
            continue
        bin_indices = np.arange(end - start) + start
        mean = np.sum(bin_indices * hist[start:end]) / prob
        var += prob * (mean - total_mean) ** 2
    return var

def find_best_thresholds(hist, classes):
    best_var = -1.0
    best_thresh = ()
    total_mean = np.sum(np.arange(len(hist)) * hist)

    thresholds_list = list(combinations(np.arange(1, len(hist)), classes - 1))

    for thresh in thresholds_list:
        var = between_class_variance(hist, thresh, total_mean)
        if var > best_var:
            best_var = var
            best_thresh = thresh

    return best_thresh

def multi_otsu(image, classes=3, levels=64):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3:
        raise ValueError("Input must be a grayscale image")

    # Step 1: compute reduced histogram
    hist = compute_histogram(image, levels=levels)
    
    # Step 2: find thresholds in reduced bin space
    thresholds_coarse = find_best_thresholds(hist, classes)

    # Step 3: scale thresholds back to 0–255 range
    scale = 256 // levels
    thresholds = [t * scale for t in thresholds_coarse]

    # Step 4: segment image using final thresholds
    regions = np.digitize(image, bins=thresholds)
    return regions, thresholds

def multi_otsu_ski(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholds = threshold_multiotsu(image, classes=3)
    regions = np.digitize(image, bins=thresholds)
    return regions, thresholds

def test():
    image = cv2.imread('CV/Segmentation/images/bobama.jpg')
    
    start_time = time.time()
    regions, _ = multi_otsu(image, classes=3, levels=64)
    end_time = time.time()
    print(f"Mean spectral Time: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    regions_ski, _ = multi_otsu_ski(image)
    end_time = time.time()
    print(f"OpenCV spectral Time: {end_time - start_time:.2f} seconds")

    cv2.imshow('scratch', regions.astype(np.uint8) * 85) 
    cv2.imshow('ski', regions_ski.astype(np.uint8) * 85)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test()