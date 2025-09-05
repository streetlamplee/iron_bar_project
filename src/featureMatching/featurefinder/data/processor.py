import cv2
import numpy as np
from glob import glob
import os

def processor():
    image_list = glob("./origin/*.jpg")
    for iname in image_list:
        image = cv2.imread(iname)
        if image.shape[:2] != (1080, 1920):
            image1 = image[0:1080, 0:1920]
            image2 = image[0:1080, 1920:3840]
            image3 = image[1080:2160, 0:1920]
            image4 = image[1080:2160, 1920:3840]

            cv2.imwrite(iname.replace(".jpg", f"_1.jpg"), image1)
            cv2.imwrite(iname.replace(".jpg", f"_2.jpg"), image2)
            cv2.imwrite(iname.replace(".jpg", f"_3.jpg"), image3)
            cv2.imwrite(iname.replace(".jpg", f"_4.jpg"), image4)

            os.remove(iname)
        else:
            pass


if __name__ == "__main__":
    processor()