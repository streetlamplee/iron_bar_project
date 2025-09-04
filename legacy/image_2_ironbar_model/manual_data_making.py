import cv2
import numpy as np

class InteractiveLineDrawer:
    def __init__(self, image, image_size=(256, 256), window_name="Image"):
        self.window_name = window_name
        self.image = image
        self.image_size = image_size
        self.display_image = self.image.copy()
        self.current_pos = (0, 0)
        self.mode = 'horizontal'
        self.horizontal_lines = []
        self.vertical_lines = []

    def mouse_event(self, event, x, y, flags, param):
        self.current_pos = (x, y)
        self.display_image = self.image.copy()

        if self.mode == 'horizontal':
            cv2.line(self.display_image, (0, y), (self.display_image.shape[1], y), (0, 0, 255), 1)
        else:
            cv2.line(self.display_image, (x, 0), (x, self.display_image.shape[0]), (255, 0, 0), 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'horizontal':
                self.horizontal_lines.append(y)
                cv2.line(self.image, (0, y), (self.image.shape[1], y), (0, 0, 255), 1)
            else:
                self.vertical_lines.append(x)
                cv2.line(self.image, (x, 0), (x, self.image.shape[0]), (255, 0, 0), 1)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mode = 'vertical' if self.mode == 'horizontal' else 'horizontal'

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_event)

        while True:
            temp = self.display_image.copy()
            if self.mode == 'horizontal':
                cv2.line(temp, (0, self.current_pos[1]), (temp.shape[1], self.current_pos[1]), (0, 0, 255), 1)
            else:
                cv2.line(temp, (self.current_pos[0], 0), (self.current_pos[0], temp.shape[0]), (255, 0, 0), 1)

            cv2.imshow(self.window_name, temp)
            key = cv2.waitKey(1)

            if key in [13, 27]:  # Enter or ESC
                break

        cv2.destroyAllWindows()
        return self.horizontal_lines, self.vertical_lines

class clickHandler:
    def __init__(self):
        self.clicked_points = []
    def handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append([x,y])

def clicked_point_finder(image:np.ndarray):
    click = clickHandler()
    image_copy = image.copy()
    cv2.imshow("Please click the Crossed point", image_copy)
    cv2.setMouseCallback("Please click the Crossed point", click.handler)
    while True:
        key = cv2.waitKey(1)
        if len(click.clicked_points) != 0:
            cv2.circle(image_copy, click.clicked_points[-1], radius=3, thickness=-1, color=(0, 0, 255))
        cv2.imshow("Please click the Crossed point", image_copy)

        if key == 13:
            break
    cv2.destroyWindow('Please click the Crossed point')
    return click.clicked_points