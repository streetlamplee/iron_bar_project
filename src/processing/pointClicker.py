import cv2

class PointClicker:
    def __init__(self, num_point, window_name="Click Points"):
        self.window_name = window_name
        self.points = []
        self.max_points = num_point
        self.multiplier = 1.0
        self.done = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.max_points:
            x_mul = int(x * self.multiplier)
            y_mul = int(y * self.multiplier)
            self.points.append((x_mul, y_mul))
            # 빨간 점 표시
            cv2.circle(self.image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
            cv2.imshow(self.window_name, self.image)

    def get_points(self, image, ratio = '16:9'):
        self.points = []
        if image.shape[0] > 900:
            self.multiplier = image.shape[0] / 900
            if ratio == '4:3':
                image = cv2.resize(image ,(1200, 900))
            elif ratio == '16:9':
                image = cv2.resize(image, (1600, 900))
            else:
                raise 'ratio value must be one of "4:3" or "16:9"'
        self.image = image.copy()

        cv2.imshow(self.window_name, self.image)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            key = cv2.waitKey(1)

            if key == 13:  # Enter 키
                if len(self.points) == self.max_points:
                    cv2.destroyWindow(self.window_name)
                    return self.points
                else:
                    print(f"점 {self.max_points}개를 모두 클릭해야 합니다. 현재: {len(self.points)}개")
            elif key == 27:  # ESC 키
                cv2.destroyWindow(self.window_name)
                raise Exception("user pause")
