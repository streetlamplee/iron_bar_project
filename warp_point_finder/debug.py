import detect_marker
import getCameraPositionFromMarker
import prjection
import numpy as np
import cv2

def main():
    """
    해당 함수는 main에 사용하기 전 테스트한 함수입니다.
    불러다가 쓰지 마세요.
    """
    idx = 0
    _, k, d, rvec, tvec = detect_marker.detectMarker("./raspi_image_with_marker.jpg", "./images", idx)

    input0 = np.array([0.,0.,0.]).reshape(1,1,3)
    input1 = np.array([-0.5, 0.5, 0]).reshape(1, 1, 3)
    input2 = np.array([0.5, 0.5, 0]).reshape(1, 1, 3)
    input3 = np.array([0.5, -0.5, 0]).reshape(1, 1, 3)
    input4 = np.array([-0.5, -0.5, 0]).reshape(1, 1, 3)

    point_2d_0 = prjection.project(input0, k, d, rvec, tvec)
    point_2d_1 = prjection.project(input1, k, d, rvec, tvec)
    point_2d_2 = prjection.project(input2, k, d, rvec, tvec)
    point_2d_3 = prjection.project(input3, k, d, rvec, tvec)
    point_2d_4 = prjection.project(input4, k, d, rvec, tvec)

    print(f"Point 1 (0,0,0): {point_2d_1}")
    print(f"Point 2 (0.01,0,0): {point_2d_2}")

    from extension import CamArrayIdx
    mask = CamArrayIdx(idx)
    canvas = cv2.imread("raspi_image_with_marker.jpg")
    canvas = canvas[mask[0]:mask[1], mask[2]:mask[3]]

    canvas = cv2.circle(canvas, point_2d_0, radius=5, thickness=-1, color=(0, 0, 255))
    canvas = cv2.circle(canvas, point_2d_1, radius=5, thickness=-1, color=(0, 0, 255))
    canvas = cv2.circle(canvas, point_2d_2, radius=5, thickness=-1, color=(0, 0, 255))
    canvas = cv2.circle(canvas, point_2d_3, radius=5, thickness=-1, color=(0, 0, 255))
    canvas = cv2.circle(canvas, point_2d_4, radius=5, thickness=-1, color=(0, 0, 255))


    from extension import image_show
    image_show(canvas)
    cv2.imwrite("./1cm_diff_image.jpg", canvas)

if __name__ == "__main__":
    main()