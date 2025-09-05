import random
import numpy as np
import cv2

def randomseed(x : int):
    # Python 내장 random 모듈 시드 설정
    random.seed(x)

    # NumPy 시드 설정
    np.random.seed(x)

    # OpenCV의 RNG 시드 설정
    cv2.setRNGSeed(x)
