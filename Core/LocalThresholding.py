import cv2 as cv
import numpy as np
from Core.OptimalThreshold import iterative_threshold
from Core.otsu_thresholding import otsu_threshold

def local_optimal_thresholding(img, threshold_type, patch_size=64):
    h, w = img.shape
    result = np.zeros_like(img)

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]

            if patch.size == 0:
                continue
            
            # Thresholding types
            if threshold_type == 'optimal':
                threshold_patch = iterative_threshold(patch, 0.5, 255)
            elif threshold_type == 'otsu':
                threshold_patch = otsu_threshold(patch)[1]
            elif threshold_type == 'spectral':
                pass

            result[y:y+patch_size, x:x+patch_size] = threshold_patch

    return result
