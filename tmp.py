import cv2
import numpy as np

img = cv2.imread('250514/iter_5_basis_seg_image.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold_value = np.sum(np.where(img > 255 / 2, img, 0)) / np.count_nonzero(img > 255 / 2)
print(threshold_value)

img = np.where(img > threshold_value, 255, 0)
img = img.astype(np.uint8)

cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyWindow('')