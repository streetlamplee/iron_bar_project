import cv2
import numpy as np

class MouseSquareDrawer:
    def __init__(self, image, square_size=256):
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
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
            self.mouse_inside = True

        elif event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: ({x}, {y})")
            self.clicked_points.append((x, y))

    def draw_square(self, img):
        if self.mouse_inside:
            top_left = (self.mouse_x, self.mouse_y)
            bottom_right = (
                min(self.mouse_x + self.square_size, img.shape[1]),
                min(self.mouse_y + self.square_size, img.shape[0])
            )
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    def run(self):
        while True:
            frame = self.canvas.copy()  # 원본 유지
            self.draw_square(frame)     # 마우스 위치에 정사각형만 덧그림

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC 키로 종료
                break

        cv2.destroyAllWindows()

    def get_clicked_points(self):
        return self.clicked_points
