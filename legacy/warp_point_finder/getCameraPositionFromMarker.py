import cv2
import numpy as np

def get_camera_pose_from_marker(rvec, tvec):
    """
    rvec, tvec: 마커의 카메라 좌표계 pose
    반환: 카메라의 마커 좌표계 pose (R, t)
    """
    R_cm, _ = cv2.Rodrigues(rvec)        # 마커의 회전행렬 in 카메라 좌표계
    R_mc = R_cm.T                        # 역행렬 (transpose)
    t_mc = -R_mc @ tvec.reshape(3, 1)    # 위치 역변환

    return R_mc, t_mc
