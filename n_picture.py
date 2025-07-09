import cv2
import numpy as np

class Picture():
    def __init__(self, data_dict:dict):
        self._picture = cv2.cvtColor(cv2.imread(data_dict["image_path"]), cv2.COLOR_BGR2RGB)
        self._picture = cv2.resize(self._picture, (1360, 1020), interpolation = cv2.INTER_LINEAR)
        self._cam_position = np.array(data_dict["cam_position"])
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
        return self._cam_position

    @property
    def rvec(self):
        return self._rvec

    @rvec.setter
    def rvec(self, value):
        self._rvec = value

    @property
    def tvec(self):
        return self._tvec

    @tvec.setter
    def tvec(self, value):
        self._tvec = value