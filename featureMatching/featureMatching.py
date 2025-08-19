import cv2
import numpy as np
from extension import image_show, CamArrayIdx

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

    sift = cv2.SIFT.create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann_idx_kdtree = 1
    idx_params = dict(algorithm = flann_idx_kdtree, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(idx_params,search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []

    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 30
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

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
    if type(points) != np.ndarray:
        points = np.float32(points)
    if points.dtype != np.float32 and points.dtype != np.float64:
        points = np.float32(points)

    points = points.reshape(-1, 1, 2)

    result = cv2.perspectiveTransform(points, M)

    return result

if __name__ == "__main__":
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