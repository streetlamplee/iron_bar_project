import cv2
import numpy as np
from tqdm import tqdm
from n_cache_manager import cache_manager
import os

class Camera():
    def __init__(self, picture_path):
        cache = cache_manager()
        cache.load_cache()
        if "camera_matrix" not in cache.cache.keys() and "dist_coeffs" not in cache.cache.keys():
            self.ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = self.calibrate_camera_from_chessboards(picture_path)

            print(f"재투영 오차: {self.ret}")
            print(f"카메라 intrinsic mat : ")
            print(self.camera_matrix)
            print(f"왜곡 계수: {self.dist_coeffs}")
            cache.set_cache("ret", self.ret)
            cache.set_cache("camera_matrix", self.camera_matrix.tolist())
            cache.set_cache("dist_coeffs", self.dist_coeffs.tolist())
            cache.save_cache()

        else:
            self.ret = True
            self.camera_matrix = cache.get_cache("camera_matrix")
            self.dist_coeffs = cache.get_cache("dist_coeffs")


    def calibrate_camera_from_chessboards(self, image_paths:str, pattern_size = (9,6), square_size = 24, display = False):
        """
            체스보드 이미지를 이용해 카메라 캘리브레이션 수행

            Args:
                image_paths (list): 체스보드 이미지 파일 경로 리스트.
                pattern_size (tuple): 내부 코너 수 (cols, rows) 예: (7, 7)
                square_size (float): 체스보드 셀의 실제 크기 (예: 30.0, 단위: mm 또는 cm 등)
                display (bool): 각 이미지에서 검출된 코너를 시각화할지 여부.

            Returns:
                ret: 캘리브레이션 결과 (재투영 오차)
                camera_matrix: 카메라 내부 파라미터 행렬
                dist_coeffs: 왜곡 계수
                rvecs: 각 이미지의 회전 벡터 목록
                tvecs: 각 이미지의 평행 이동 벡터 목록
            """
        # 3D 객체 점 준비 (체스보드가 z=0 평면에 있다고 가정)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp = objp * square_size  # 실제 크기로 스케일링

        # 여러 이미지에서의 3D 점과 2D 이미지 점 저장 리스트
        objpoints = []  # 3D 점 (각 이미지마다 동일)
        imgpoints = []  # 2D 점 (각 이미지마다 다름)

        image_list = os.listdir(image_paths)
        image_list = [f"{image_paths}/{x}" for x in image_list]

        # 체스보드 코너 검출 및 저장
        for fname in tqdm(image_list, desc="Looking for Corner of Chessboard"):
            img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            if img is None:
                print(f"이미지를 불러올 수 없습니다: {fname}")
                continue
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                # 코너 위치 서브픽셀 보정
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                objpoints.append(objp)
                imgpoints.append(corners)

                if display:
                    img_drawn = cv2.drawChessboardCorners(img.copy(), pattern_size, corners, ret)
                    cv2.imshow("Detected Chessboard Corners", img_drawn)
                    cv2.waitKey(1000)
            else:
                print(f"체스보드 코너 검출 실패: {fname}")

        if display:
            cv2.destroyAllWindows()

        # 캘리브레이션 수행: 이미지 크기를 gray.shape[::-1]로 전달
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        return ret, camera_matrix, dist_coeffs, rvecs, tvecs

    def detect_chessboard_corners(image, pattern_size, display=False):
        """
        입력 이미지에서 체스보드 내부 코너를 검출하는 함수.

        Args:
            image (np.array): 입력 이미지 (컬러 또는 그레이스케일).
            pattern_size (tuple): 내부 코너 수 (열, 행). 예를 들어, (7, 7) 또는 (9, 6).
            display (bool): True인 경우, 검출된 코너를 그린 이미지를 화면에 출력.

        Returns:
            ret (bool): 체스보드 코너 검출 성공 여부.
            corners (np.array or None): 검출된 코너 좌표 (float32, [N x 1 x 2] shape) 또는 검출 실패 시 None.
        """

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            if display:
                img_corners = cv2.drawChessboardCorners(image.copy(), pattern_size, corners, ret)
                cv2.imshow("Chessboard Corners", img_corners)
                cv2.waitKey(0)
                cv2.destroyWindow("Chessboard Corners")
        else:
            print("Couldn't Find Chessboard Corners")
            corners = None
        return ret, corners