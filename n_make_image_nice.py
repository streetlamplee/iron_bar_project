import cv2
import numpy as np
from extension import array_norm

def custom_image_thresholding(image, kernel, threshold):
    b_image = (image > 255. * 2. / 3.).astype(np.uint8)

    response = cv2.filter2D(b_image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    denometer = kernel.shape[0] * kernel.shape[1]
    output = np.where(response >= threshold, response / denometer, 0)

    output = array_norm(output) * 255

    output = output.astype(np.uint8)

    return output