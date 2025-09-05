import numpy as np
import cv2

def get_projection_matrix(camera_matrix, rvec, tvec):
    """
    camera_matrix: 3x3 intrinsic matrix (K)
    rvec: 3x1 rotation vector
    tvec: 3x1 translation vector

    return: 3x4 projection matrix
    """
    # rvec (Rodrigues 회전 벡터) → R (3x3 회전 행렬)
    R, _ = cv2.Rodrigues(rvec)

    # [R | t] 형태의 3x4 extrinsic matrix 생성
    Rt = np.hstack((R, tvec))

    # Projection matrix = K [R | t]
    P = camera_matrix @ Rt

    return P

def project(p_3d, k, d, rvec,tvec):
    p_3d = np.reshape(p_3d, (1,1,3))

    p_2d, _ = cv2.projectPoints(
        p_3d,
        rvec,
        tvec,
        k,
        d
    )

    x, y = p_2d[0,0] # p_2d.shape == (1,1,2) 이므로

    return (int(x),int(y))