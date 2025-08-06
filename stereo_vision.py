import cv2
import numpy as np
from extension import image_show

def stereo_vision():
    image_L = cv2.imread("./raspi_image.jpg")[0:1080, 0:1920].astype(np.uint8)
    image_R = cv2.imread("./raspi_image.jpg")[0:1080, 1920:3840].astype(np.uint8)

    image_L = cv2.cvtColor(image_L, cv2.COLOR_BGR2GRAY)
    image_R = cv2.cvtColor(image_R, cv2.COLOR_BGR2GRAY)

    image_show(image_L)
    image_show(image_R)

    stereo = cv2.StereoBM.create(numDisparities=1024, blockSize=21)

    disparity = stereo.compute(image_L, image_R)
    disparity = cv2.resize(disparity, (960,540))
    image_show(disparity)

if __name__ == "__main__":
    stereo_vision()

4