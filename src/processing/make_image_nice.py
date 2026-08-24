"""
합성 결과에서 특정 패턴(kernel)에 해당하는 영역만 남기는 실험용 thresholding.
현재 메인 파이프라인에서는 사용하지 않는다.
"""

import cv2
import numpy as np
from extension import array_norm

def custom_image_thresholding(image, kernel, threshold):
    """
    이진화한 이미지에 kernel을 convolution 해서, 응답이 threshold 이상인 곳만 남긴다.

    :param image: 입력 이미지 (0~255)
    :param kernel: 찾고자 하는 패턴을 나타내는 커널
    :param threshold: 살아남기 위해 필요한 최소 응답값 (커널과 겹친 픽셀 수)
    :return: 0~255로 정규화된 결과 (uint8)
    """
    # 중간값(127.5)을 기준으로 0/1 이진화
    b_image = (image > 255. * 1. / 2.).astype(np.uint8)

    # 커널과 겹치는 픽셀 수 = 해당 위치에서의 패턴 일치 정도
    response = cv2.filter2D(b_image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # 커널 넓이로 나눠 0~1 비율로 만들고, 기준 미달인 곳은 0으로 버린다.
    denometer = kernel.shape[0] * kernel.shape[1]
    output = np.where(response >= threshold, response / denometer, 0)

    # 남은 값들을 0~255 범위로 다시 펴서 눈으로 볼 수 있게 만든다.
    output = array_norm(output) * 255

    output = output.astype(np.uint8)

    return output
