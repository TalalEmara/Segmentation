import cv2
import numpy as np

def local_thresholding(image, window_size=11, C=9):
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer. Try again, wise one!")
    
    local_mean = cv2.GaussianBlur(image, (window_size, window_size), 3)
    thresholds = local_mean - C
    binary = (image > thresholds).astype(np.uint8) * 255
    return binary


def localThresholdingCV(image, window_size=11, C=9):
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer. Try again, wise one!")

    # image = cv2.GaussianBlur(image, (11, 11), 0)

    thresh = cv2.adaptiveThreshold(image, 255, 
                               cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 
                               window_size, C)
    return thresh

def test():
    image = cv2.imread("CV/Segmentation/images/Snake/Coins04.jpg", 0)  # Load a grayscale image
    local_thresholded_image_cv = localThresholdingCV(image, window_size=11, C=9)
    local_thresholded_image = local_thresholding(image, window_size=11, C=9)
    cv2.imshow("Local Thresholding (custom)", local_thresholded_image)
    cv2.imshow("Local Thresholding (opencv)", local_thresholded_image_cv)
    cv2.waitKey(0)

test()