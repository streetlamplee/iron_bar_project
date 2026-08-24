"""
segmentation 결과를 다중 시점으로 합성하기 전에 적용하는 전처리 blur.
"""

import cv2
import numpy as np

def custom_blur(img:np.ndarray, iteration = 1):
    """
    가늘게 끊긴 철근 마스크를 두껍고 매끄럽게 만든다.

    여러 시점의 마스크를 겹칠 때 좌표가 미세하게 어긋나도 서로 겹치도록,
    선을 의도적으로 굵히고 경계를 흐리게 만드는 것이 목적이다.

    :param img: segmentation 결과 (0~255 grayscale)
    :param iteration: 팽창/침식 반복 횟수. 클수록 선이 더 두꺼워진다.
    :return: 두꺼워지고 흐려진 마스크 (uint8)
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)

    # dilate를 erode보다 1회 더 수행한다 -> closing(끊긴 선 잇기) + 선을 한 단계 더 굵게.
    img = cv2.dilate(img, kernel, iterations=iteration+1)
    img = cv2.erode(img, kernel, iterations=iteration)

    img_median = cv2.medianBlur(img, 5)
    img_Gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=5, sigmaY=5)
    # 원본과 흐린 버전 중 밝은 값을 취해, 선을 잃지 않으면서 주변으로 번지게 한다.
    # 주의: np.maximum의 세 번째 인자는 out 파라미터라 img_median은 비교 대상이 아니라
    #       결과를 덮어쓰는 버퍼로 쓰인다 (실질적으로 img 와 img_Gaussian 의 max).
    img_blur = np.maximum(img, img_Gaussian, img_median)

    return img_blur
