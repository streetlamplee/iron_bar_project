import cv2
import numpy as np

def triangluate(mat1, rv1, tv1, mat2, rv2, tv2, point1, point2):

    # rv tv를 이용해 먼저 Projection Matrix 생성

    R1, _ = cv2.Rodrigues(rv1)
    R2, _ = cv2.Rodrigues(rv2)

    Rt1 = np.hstack((R1, np.reshape(tv1, (3, 1))))
    Rt2 = np.hstack((R2, np.reshape(tv2, (3, 1))))

    P1 = mat1 @ Rt1
    P2 = mat2 @ Rt2

    # point를 전처리
    point1 = np.array(point1, dtype = np.int16)
    point2 = np.array(point2, dtype = np.int16)

    point1 = np.reshape(point1, (2,1))
    point2 = np.reshape(point2, (2,1))

    # Projection Matrix를 이용해서 삼각측량

    points4D = cv2.triangulatePoints(P1, P2, point1, point2)
    points3D = points4D[:3] / points4D[3]

    return points3D

if __name__ == "__main__":
    import detect_marker

    r, m, rv, tv = detect_marker.orderPoint("./marker_image_4.jpg", "./images")

    img0 = [m[0], rv[0], tv[0]]
    img1 = [m[1], rv[1], tv[1]]
    img2 = [m[2], rv[2], tv[2]]
    img3 = [m[3], rv[3], tv[3]]

    args = [*img0, *img1, [0, 0], [617, 854]]
    print(triangluate(*args))

    from getCameraPositionFromMarker import get_camera_pose_from_marker
    R, t = get_camera_pose_from_marker(img0[1], img0[2])
    print(t.ravel())
    R, t = get_camera_pose_from_marker(img1[1], img1[2])
    print(t.ravel())
    R, t = get_camera_pose_from_marker(img2[1], img2[2])
    print(t.ravel())
    R, t = get_camera_pose_from_marker(img3[1], img3[2])
    print(t.ravel())


