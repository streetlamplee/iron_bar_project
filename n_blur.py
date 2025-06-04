import cv2
import numpy as np

def custom_blur(img:np.ndarray, iteration = 1):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    img_bilateral = cv2.bilateralFilter(img, -1, 50, 15)
    img_median = cv2.medianBlur(img, 7)
    img_Gaussian = cv2.GaussianBlur(img, (0,0), sigmaX = 5, sigmaY = 5)
    img_blur = np.maximum(img, img_Gaussian, img_median)


    for _ in range(iteration):
        img = cv2.dilate(img_blur, kernel, iterations=1)

        img_median = cv2.medianBlur(img, 7)
        img_Gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=5, sigmaY=5)
        img_blur = np.maximum(img, img_Gaussian, img_median)

    return img_blur