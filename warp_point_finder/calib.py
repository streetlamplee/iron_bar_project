import cv2
import numpy as np
import os
import glob
from extension import image_show, CamArrayIdx

'''
@brief ./warp_point_finder/images 폴더 내부의 jpg 파일 중, 좌상단의 사진에서 체스보드를 탐지합니다.
@param folderName 체스보드 이미지 폴더의 경로
@param idx CamArray 중 몇 번째 사진을 이용할 것인지 (0: 좌상단, 1: 우상단, 2: 좌하단, 3: 우하단)
@return 해당 camera의 calibrateCamera 함수 return 값
'''
def findChessboard(folderName, idx):
    mask = CamArrayIdx(idx)
    CHECKERBOARD = (9,6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objPoints = []
    imgPoints = []

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
    prevImageShape = None

    images = glob.glob(folderName + "/*.jpg")
    for fname in images:
        img = cv2.imread(fname)
        img = img[mask[0]:mask[1], mask[2]:mask[3]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray,
                                                 CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objPoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgPoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # image_show(img, 'img')

    h,w = img.shape[:2]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs



