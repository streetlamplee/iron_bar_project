import numpy as np
import cv2
import glob
from tqdm import tqdm


def create_chessboard(num_cols, num_rows, cell_size):
    """
    체스보드 패턴 생성 함수
    인자:
      num_cols: 체스보드의 열(가로) 칸 수
      num_rows: 체스보드의 행(세로) 칸 수
      cell_size: 한 셀의 크기 (픽셀 단위, 정사각형의 한 변 길이)
    반환:
      chessboard: 0과 255를 가진 uint8 타입의 체스보드 이미지 (np.array)
    """
    # 전체 이미지 크기 계산
    width = num_cols * cell_size
    height = num_rows * cell_size

    # 빈 체스보드 (초기값: 0으로 검정색)
    chessboard = np.zeros((height, width), dtype=np.uint8)

    # 각 셀에 대해 색상 지정 (0: 검정, 255: 흰색)
    for row in range(num_rows):
        for col in range(num_cols):
            # (row, col)의 셀은 흰색으로 채울지 여부 결정
            # 일반적으로 (row + col)이 짝수이면 흰색, 홀수이면 검정색
            if (row + col) % 2 == 0:
                # 해당 셀 영역 채우기
                chessboard[row * cell_size:(row + 1) * cell_size,
                col * cell_size:(col + 1) * cell_size] = 255
    return chessboard

def detect_chessboard_corners(image, pattern_size, display = False):
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
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)

        if display:
            img_corners = cv2.drawChessboardCorners(image.copy(), pattern_size, corners, ret)
            cv2.imshow("Chessboard Corners", img_corners)
            cv2.waitKey(0)
            cv2.destroyWindow("Chessboard Corners")
    else:
        print("Couldn't Find Chessboard Corners")
        corners = None
    return ret, corners

def calibrate_camera_from_chessboards(image_paths, pattern_size, square_size, display=False):
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

    # 체스보드 코너 검출 및 저장
    for fname in tqdm(image_paths, desc="Looking for Corner of Chessboard"):
        img = fname
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

# if __name__ == '__main__':
    # board = create_chessboard(7, 7, 30)
    # pattern_size = (6,6)  # 예: 7 x 7 내부 코너
    # found, corners = detect_chessboard_corners(board, pattern_size, display=True)
    #
    # if found:
    #     print("체스보드 코너 검출 성공!")
    #     print("검출된 코너 좌표:\n", corners)
    # else:
    #     print("체스보드 코너 검출 실패!")



    # # calibrate_camera_from_chessboards 테스트
    # image_paths = glob.glob("*.png")
    # pattern_size = (7, 6)  # 내부 코너 7x7
    # square_size = 30.0  # 각 셀의 실제 크기 30mm (단위는 일관되게 사용)
    #
    # ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_from_chessboards(
    #     image_paths, pattern_size, square_size, display=True
    # )
    #
    # print("재투영 오차:", ret)
    # print("내부 파라미터 (카메라 행렬):\n", camera_matrix)
    # print("왜곡 계수:\n", dist_coeffs)

    pass