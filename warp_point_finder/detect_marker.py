import cv2
from calib import findChessboard
from extension import image_show, CamArrayIdx
import numpy as np


'''
@brief param으로 받은 이미지 내에서, aruco marker를 찾습니다.
@param fname marker_detection을 진행할 이미지
@param folderName findChessboard 함수의 param
@param idx CamArray 중 몇 번째 사진을 이용할 것인지 (0: 좌상단, 1: 우상단, 2: 좌하단, 3: 우하단)
@return result: key = id, value = markercenter인 dict
@return k : 카메라 내부행렬
@return d : 카메라 왜곡행렬
@return target_rvec : target인 id를 가지는 marker 기준, 카메라의 rvec (마커가 1개인 경우, 자동으로 그 marker가 기준)
@return target_tvec : target인 id를 가지는 marker 기준, 카메라의 tvec (마커가 1개인 경우, 자동으로 그 marker가 기준)
'''
def detectMarker(fname, folderName, idx):
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }


    mask = CamArrayIdx(idx)

    _, k, d, rv, tv = findChessboard(folderName, idx)

    image = cv2.imread(fname)
    image = image[mask[0]:mask[1], mask[2]:mask[3]]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_5X5_100"])
    arucoParams = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)

    result = [0,0,0,0]
    target_marker_idx = 1
    if len(corners) > 0:
        for num, (i, c) in enumerate(zip(ids, corners)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(c, 0.07, k,d)
            if i == target_marker_idx or len(corners) == 1:
                target_rvec = rvec
                target_tvec = tvec
            cv2.aruco.drawDetectedMarkers(image, corners)

            cv2.drawFrameAxes(image, k, d, rvec, tvec, 0.01)
            cv2.putText(image, f"{i}", c[0][0].astype(np.int16), cv2.FONT_HERSHEY_SIMPLEX, color = (0,0,255), fontScale=2)

            markerCenter = np.reshape(c, (4, 2))
            markerCenter = np.mean(markerCenter, axis=0)
            markerCenter = markerCenter.astype(np.int16)

            result[num] = (int(i), markerCenter)

        image_show(image)

    return result, k, d, target_rvec, target_tvec
'''
@brief detectMarker 수행 후, aruco Marker ID 별로 이미지 내의 좌표를 반환하는 함수
@param detectMarker 함수의 param을 그대로 사용
@return
'''
def orderPoint(fname, folderName):
    image1, matrix1, _, rv1, tv1 = detectMarker(fname, folderName, 0)
    image2, matrix2, _, rv2, tv2 = detectMarker(fname, folderName, 1)
    image3, matrix3, _, rv3, tv3 = detectMarker(fname, folderName, 2)
    image4, matrix4, _, rv4, tv4 = detectMarker(fname, folderName, 3)
    images = [image1, image2, image3, image4]
    matrixs = [matrix1, matrix2, matrix3, matrix4]
    rvs = [rv1, rv2, rv3, rv4]
    tvs = [tv1, tv2, tv3, tv4]
    # result will be ordered by image1, image2, ... , image4
    result = dict()
    keys = []
    for i in range(len(images[0])):
        keys.append(images[0][i][0])
    for k in keys:
        result[k] = list()

    for image in images:
        for id_center in image:
            id, center = id_center
            if id == keys[0]:
                result[id].append(center)
            elif id == keys[1]:
                result[id].append(center)
            elif id == keys[2]:
                result[id].append(center)
            elif id == keys[3]:
                result[id].append(center)
            else:
                raise "Not valid Id is returned by 'detectMarker'"

    return result, matrixs, rvs, tvs

if __name__ == "__main__":
    # res, _, _, _ = orderPoint("./marker_image_4.jpg", "./images")
    #
    # print(res)
    detectMarker("./raspi_image_with_marker.jpg", "./images", 1)