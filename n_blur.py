import cv2
import numpy as np

def custom_blur(img:np.ndarray):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    img_bilateral = cv2.bilateralFilter(img, -1, 50, 15)
    img_median = cv2.medianBlur(img, 7)
    img_Gaussian = cv2.GaussianBlur(img, (0,0), 3)

    img_blur = np.maximum(img, img_Gaussian)

    return img_blur