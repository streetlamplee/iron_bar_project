"""
사진 한 장과 그 사진을 찍은 카메라의 자세(pose)를 함께 들고 다니기 위한 컨테이너.

3D 좌표를 각 카메라 화면에 투영해 warp point를 구하려던 시도에서 사용했다.
현재 메인 파이프라인에서는 사용하지 않는다.
"""

import cv2
import numpy as np

class Picture():
    def __init__(self, data_dict:dict):
        """
        :param data_dict: {"image_path": 사진 경로, "cam_position": 카메라의 3D 위치} 형태의 dict
        """
        # 모델/표시 규약에 맞춰 RGB로 변환하고, 처리 비용을 줄이기 위해 고정 크기로 축소한다.
        self._picture = cv2.cvtColor(cv2.imread(data_dict["image_path"]), cv2.COLOR_BGR2RGB)
        self._picture = cv2.resize(self._picture, (1360, 1020), interpolation = cv2.INTER_LINEAR)
        self._cam_position = np.array(data_dict["cam_position"])
        # rvec/tvec(카메라의 회전/이동)은 생성 시점에 알 수 없고,
        # 마커 검출이나 캘리브레이션을 거친 뒤 외부에서 채워 넣는다.
        self._rvec = None
        self._tvec = None

    @property
    def picture(self):
        return self._picture

    @picture.setter
    def picture(self, value):
        self._picture = value

    @property
    def cam_position(self):
        return self._cam_position

    @cam_position.setter
    def cam_position(self, value):
        # 주의: 값을 저장하지 않고 기존 값을 반환만 한다. setter로 동작하지 않는다.
        return self._cam_position

    @property
    def rvec(self):
        """카메라의 회전 벡터 (Rodrigues 표현)"""
        return self._rvec

    @rvec.setter
    def rvec(self, value):
        self._rvec = value

    @property
    def tvec(self):
        """카메라의 이동 벡터"""
        return self._tvec

    @tvec.setter
    def tvec(self, value):
        self._tvec = value
