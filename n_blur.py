import cv2
import numpy as np

def custom_blur(img:np.ndarray, iteration = 1):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)

    img = cv2.dilate(img, kernel, iterations=iteration+1)
    img = cv2.erode(img, kernel, iterations=iteration)

    img_median = cv2.medianBlur(img, 5)
    img_Gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=5, sigmaY=5)
    img_blur = np.maximum(img, img_Gaussian, img_median)

    return img_blur