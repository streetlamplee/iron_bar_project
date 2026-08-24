"""
관심 영역(4점)을 정사각 평면으로 펴는 perspective warp 유틸.

서로 다른 각도에서 찍은 사진들을 같은 좌표계 위에 올리기 위해 사용한다.
"""

import numpy as np
from tqdm.utils import disp_trim

from processing.camera import Camera
import cv2



def warp_perspective(image:np.ndarray, target_points:np.ndarray, dst = np.float32([[0,0],[1024,0],[1024,1024],[0,1024]])):
    """
    원본 이미지의 네 점(target_points)을 dst 의 네 점으로 옮기는 perspective 변환을 적용한다.

    :param image: 변환할 이미지. grayscale/컬러 모두 가능.
    :param target_points: 원본 이미지 위의 4점. float32 여야 하며,
                          순서는 dst 와 반드시 1:1로 대응해야 한다 (보통 좌상 -> 우상 -> 우하 -> 좌하).
    :param dst: 목적지 평면의 4점. 기본값은 1024x1024 정사각형.
    :return: (dst 크기)로 펴진 이미지
    """
    # 4점 -> 4점 대응으로 3x3 homography 행렬 M을 구한다.
    M = cv2.getPerspectiveTransform(target_points, dst)
    # 출력 크기는 dst 좌표의 가로/세로 범위에서 그대로 계산한다.
    size_x = int( np.max(dst[:,0]) - np.min(dst[:,0]) )
    size_y = int( np.max(dst[:,1]) - np.min(dst[:,1]) )
    # WARP_FILL_OUTLIERS: 대응점이 없는 픽셀을 비워두지 않고 채운다.
    # INTER_CUBIC: 확대되는 영역의 화질을 위해 3차 보간 사용.
    warp = cv2.warpPerspective(image, M, (size_x, size_y), flags = cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

    return warp
