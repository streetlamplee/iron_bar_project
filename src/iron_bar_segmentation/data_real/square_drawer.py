"""
마우스를 따라다니는 정사각형을 그려, 잘라낼 위치를 눈으로 확인하며 고르게 해주는 도구.

make_binary.py 에서 학습 데이터 조각을 뜰 때 사용한다.
"""

import cv2
import numpy as np

class MouseSquareDrawer:
    def __init__(self, image, square_size=256):
        """
        :param image: 위치를 고를 대상 이미지 (화면 크기에 맞게 미리 축소해서 넘긴다)
        :param square_size: 표시할 정사각형 한 변의 길이. 실제로 잘라낼 크기와 맞춰야 한다.
        """
        self.canvas = image
        self.square_size = square_size
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_inside = False
        self.clicked_points = []

        self.window_name = "Mouse Tracker"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        # 마우스가 움직이면 현재 위치를 기억해 사각형을 그 자리에 그린다.
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
            self.mouse_inside = True

        # 클릭하면 그 좌표를 기록한다 (사각형의 좌상단이 된다).
        elif event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: ({x}, {y})")
            self.clicked_points.append((x, y))

    def draw_square(self, img):
        """마우스 위치를 좌상단으로 하는 사각형을 그린다. 이미지 밖으로 나가지 않게 잘라 맞춘다."""
        if self.mouse_inside:
            top_left = (self.mouse_x, self.mouse_y)
            bottom_right = (
                min(self.mouse_x + self.square_size, img.shape[1]),
                min(self.mouse_y + self.square_size, img.shape[0])
            )
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    def run(self):
        """ESC를 누를 때까지 창을 유지한다. 그동안 클릭한 좌표가 쌓인다."""
        while True:
            frame = self.canvas.copy()  # 원본 유지
            self.draw_square(frame)     # 마우스 위치에 정사각형만 덧그림

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC 키로 종료
                break

        cv2.destroyAllWindows()

    def get_clicked_points(self):
        """클릭된 좌표 목록을 돌려준다. 호출 측은 보통 첫 번째 좌표만 사용한다."""
        return self.clicked_points
