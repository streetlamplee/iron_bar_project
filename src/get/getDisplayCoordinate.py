from n_camera import Camera
import numpy as np
import cv2

def get_display_coordinate(camera:Camera, rvec, tvec, _3d_point:np.ndarray):
    '''
    3D 좌표가 카메라를 통해 촬영될 때, 이미지 상에서의 2D 위치를 반환합니다.
    이미지의 좌표계는 좌상단이 (0, 0)입니다 (OpenCV 기준).

    :param camera: 카메라 내부 파라미터(camera_matrix, dist_coeffs)를 가진 객체
    :param rvec: 회전 벡터 (Rodrigues)
    :param tvec: 이동 벡터
    :param _3d_point: (N, 3) 또는 (3,) 형태의 3D 점
    :return: (N, 2) 형태의 2D 이미지 좌표
    '''

    camera_matrix = camera.camera_matrix
    dist_coeffs = camera.dist_coeffs

    rvec = np.array(rvec, dtype = np.float32)
    tvec = np.array(tvec, dtype = np.float32)
    camera_matrix = np.array(camera_matrix, dtype = np.float32)
    dist_coeffs = np.array(dist_coeffs, dtype = np.float32)

    image_points, _ = cv2.projectPoints(_3d_point, rvec, tvec, camera_matrix, dist_coeffs)

    image_points = image_points.reshape(-1, 2)

    return image_points