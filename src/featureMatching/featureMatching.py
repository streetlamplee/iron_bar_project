"""
두 사진 사이의 특징점을 매칭해 좌표를 옮기는 모듈.

사진마다 관심 영역 4점을 일일이 클릭하지 않고,
한 장에만 찍어둔 좌표를 나머지 사진으로 자동 전달하기 위해 만들었다.
main.py 의 run_sift_warp_pipeline() 에서 사용한다.
"""

import cv2
import numpy as np
from etc.extension import image_show, CamArrayIdx

'''
@brief 이미지 2개 사이의 특징점 매칭 후, Homography Mat을 얻는 함수
@param img1 기준이 될 이미지
@param img2 대상이 될 이미지
@return retval 특징점 매칭 성공 여부
@return mat 특징점 매칭으로 얻어낸 Homography Mat
'''
def getHomographySift(img1, img2):
    if len(img1.shape) != 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) != 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT: 크기와 회전이 달라도 같은 지점을 찾아내는 특징점 추출기.
    # kp = 특징점 위치, des = 그 지점의 생김새를 숫자로 표현한 값(비교에 사용)
    sift = cv2.SIFT.create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN: 특징점 수가 많을 때 가장 비슷한 짝을 빠르게 찾아주는 근사 매칭기
    flann_idx_kdtree = 1
    idx_params = dict(algorithm = flann_idx_kdtree, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(idx_params,search_params)

    # 각 특징점마다 가장 비슷한 후보 2개를 찾는다.
    matches = flann.knnMatch(des1, des2, k=2)

    good = []

    # Lowe's ratio test: 1등이 2등보다 뚜렷하게 가까울 때만 믿을 만한 매칭으로 본다.
    # 비슷비슷한 무늬(철근이 반복되는 배경)에서 잘못 짝지어지는 것을 걸러낸다.
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 신뢰할 매칭이 이 수보다 적으면 변환 행렬을 믿을 수 없다고 보고 실패 처리한다.
    MIN_MATCH_COUNT = 30
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # RANSAC: 잘못 짝지어진 매칭이 섞여 있어도, 다수가 동의하는 변환만 채택한다.
        # 10.0은 몇 픽셀까지 오차를 허용할지의 기준값.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        matchesMask = mask.ravel().tolist()

        inlier_matches = [m for i, m in enumerate(good) if mask[i]]

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        return True, M

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return False, None


'''
@brief point 집합을 Homography Mat M에 맞추어 좌표 변환을 진행하는 함수
@param points 좌표 변환을 적용할 point의 집합
@param M 적용할 homography 행렬
@return 좌표 변환이 완료된 좌표의 집합
'''
def perspectiveTransfrom(points:list|np.ndarray, M):
    # cv2.perspectiveTransform은 float32 형식과 (N, 1, 2) 모양을 요구하므로 맞춰준다.
    if type(points) != np.ndarray:
        points = np.float32(points)
    if points.dtype != np.float32 and points.dtype != np.float64:
        points = np.float32(points)

    points = points.reshape(-1, 1, 2)

    result = cv2.perspectiveTransform(points, M)

    return result

if __name__ == "__main__":
    # 단독 실행 시 확인용: 첫 사진에 찍은 점 4개가 두 번째 사진의 어디로 옮겨지는지 그려본다.
    mask1 = CamArrayIdx(0)
    target_idx = 3
    mask2 = CamArrayIdx(target_idx)
    fname_list = ["../data_real/0818/20250818_145135.jpg",
                  "../data_real/0818/20250818_145131.jpg",
                  "../data_real/0818/20250818_145136.jpg",
                  "../data_real/0818/20250818_145140.jpg",]
    # img = cv2.imread("../raspi_image.jpg")
    # img1 = img[mask1[0]:mask1[1], mask1[2]:mask1[3]]
    # img2 = img[mask2[0]:mask2[1], mask2[2]:mask2[3]]

    img1 = cv2.imread(fname_list[0])
    img2 = cv2.imread(fname_list[1])

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # points = [[503, 1065], [559, 1393], [849, 1341], [765, 951]]
    points = [[1065, 503], [1393, 559], [1341, 849], [951, 765]]
    img1_bgr = cv2.cvtColor(img1.copy(), cv2.COLOR_GRAY2BGR)
    for p in points:
        cv2.circle(img1_bgr, p, color = (0,0,255), thickness=-1, radius=5)

    retval, M = getHomographySift(img1, img2)
    if retval:
        target_points = perspectiveTransfrom(points, M)
    else:
        target_points = []

    result = img2.copy()
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    for p in np.int32(target_points):
        p = p.reshape(2,)
        cv2.circle(result, p, color=(0, 0, 255), thickness=-1, radius=5)
    image_show(result)
    # cv2.imwrite("./target_image0.jpg", img1_bgr)
    # cv2.imwrite(f"./target_image{target_idx}.jpg",result)