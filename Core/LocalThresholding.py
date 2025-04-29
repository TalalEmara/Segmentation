import cv2 as cv
import numpy as np
from Core.OptimalThreshold import iterative_threshold
from Core.otsu_thresholding import otsu_threshold
from Core.spectral import multi_otsu

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
                # For multi-Otsu, we need to handle the multiple thresholds differently
                regions, thresholds = multi_otsu(patch, 3)
                
                # Create a thresholded image based on the regions
                threshold_patch = np.zeros_like(patch)
                for i in range(len(thresholds) + 1):
                    if i == 0:
                        mask = regions == i
                        threshold_patch[mask] = 0
                    elif i == len(thresholds):
                        mask = regions == i
                        threshold_patch[mask] = 255
                    else:
                        mask = regions == i
                        threshold_patch[mask] = thresholds[i-1]
            else:
                raise ValueError("Invalid threshold type")

            result[y:y+patch_size, x:x+patch_size] = threshold_patch

    return result

def test():
    image = cv.imread('CV/Segmentation/images/bobama.jpg')
    result = local_optimal_thresholding(image, 'spectral', patch_size=64)
    cv.imshow('scratch', result.astype(np.uint8) * 85) 
    cv.waitKey(0)
    cv.destroyAllWindows()

test()
