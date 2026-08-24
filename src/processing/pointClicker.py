"""
마우스 클릭으로 이미지 위의 좌표를 받아오는 도구.

관심 영역(warp point) 4점을 손으로 지정할 때 사용한다.
GUI 창을 띄우므로 화면이 없는 환경에서는 동작하지 않는다.
"""

import cv2

class PointClicker:
    def __init__(self, num_point, window_name="Click Points"):
        """
        :param num_point: 받을 점의 개수 (이 프로젝트에서는 4)
        :param window_name: 표시할 창 이름
        """
        self.window_name = window_name
        self.points = []            # 클릭된 좌표 (원본 해상도 기준)
        self.max_points = num_point
        self.multiplier = 1.0       # 화면 표시용으로 축소한 배율. 좌표를 원본 크기로 되돌릴 때 쓴다.
        self.done = False

    def _mouse_callback(self, event, x, y, flags, param):
        """왼쪽 버튼 클릭을 받아 좌표를 저장하고 화면에 표시한다."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.max_points:
            # 창에 보이는 좌표는 축소된 값이므로, 원본 해상도 기준으로 되돌려 저장한다.
            x_mul = int(x * self.multiplier)
            y_mul = int(y * self.multiplier)
            self.points.append((x_mul, y_mul))
            # 빨간 점 표시
            cv2.circle(self.image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
            cv2.imshow(self.window_name, self.image)

    def get_points(self, image, ratio = '16:9'):
        """
        창을 띄우고 사용자가 max_points 개를 클릭할 때까지 기다린다.

        조작: 좌클릭으로 점 찍기 -> 다 찍은 뒤 Enter로 확정 / ESC로 중단
        :param image: 점을 찍을 이미지
        :param ratio: 원본 사진의 화면비. 축소해서 보여줄 때 사용한다.
        :return: 원본 해상도 기준 좌표 리스트 [(x, y), ...]
        """
        self.points = []
        # 원본 사진이 크면 화면에 다 안 들어오므로 세로 900에 맞춰 축소해서 보여준다.
        # 이때의 축소 배율을 기억해뒀다가 클릭 좌표를 원본 크기로 복원한다.
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

        # 키 입력을 기다리는 루프. 콜백이 점을 채우는 동안 여기서 대기한다.
        while True:
            key = cv2.waitKey(1)

            if key == 13:  # Enter 키
                # 개수를 다 채운 경우에만 확정하고 반환한다.
                if len(self.points) == self.max_points:
                    cv2.destroyWindow(self.window_name)
                    return self.points
                else:
                    print(f"점 {self.max_points}개를 모두 클릭해야 합니다. 현재: {len(self.points)}개")
            elif key == 27:  # ESC 키
                # 사용자가 중단하면 예외를 던져 상위 실행을 멈춘다.
                cv2.destroyWindow(self.window_name)
                raise Exception("user pause")
