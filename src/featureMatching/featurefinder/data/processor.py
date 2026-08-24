"""
학습 데이터로 쓸 사진을 카메라별 한 장씩으로 정리하는 전처리 스크립트.
"""

import cv2
import numpy as np
from glob import glob
import os

def processor():
    """
    origin 폴더의 사진을 훑어, 4K 한 장에 4대가 붙어 있는 사진을 카메라별로 잘라 저장한다.
    이미 1920x1080인 사진은 그대로 둔다. 자른 뒤 원본은 삭제한다.
    """
    image_list = glob("./origin/*.jpg")
    for iname in image_list:
        image = cv2.imread(iname)
        # 해상도가 1920x1080이 아니면 4대가 2x2로 붙은 사진으로 간주하고 4등분한다.
        if image.shape[:2] != (1080, 1920):
            image1 = image[0:1080, 0:1920]
            image2 = image[0:1080, 1920:3840]
            image3 = image[1080:2160, 0:1920]
            image4 = image[1080:2160, 1920:3840]

            cv2.imwrite(iname.replace(".jpg", f"_1.jpg"), image1)
            cv2.imwrite(iname.replace(".jpg", f"_2.jpg"), image2)
            cv2.imwrite(iname.replace(".jpg", f"_3.jpg"), image3)
            cv2.imwrite(iname.replace(".jpg", f"_4.jpg"), image4)

            # 잘라낸 조각을 저장했으므로 원본은 지운다 (되돌릴 수 없으니 주의).
            os.remove(iname)
        else:
            pass


if __name__ == "__main__":
    processor()