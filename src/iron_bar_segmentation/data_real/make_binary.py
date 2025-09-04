import numpy as np
import cv2
import os
import extension
from iron_bar_segmentation.data_real.square_drawer import MouseSquareDrawer

Debug = True

def binary():
    os.makedirs('mask', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    now_folder = os.listdir('../../data_real/')
    now_folder = [png_file for png_file in now_folder if png_file.endswith('.png') or png_file.endswith('.jpg')]
    start_num = sorted([int(i.replace('.png','')) for i in os.listdir('data')], reverse=True)[0]
    cnt = start_num + 1
    num_target = 20
    while True:
        rand_idx = np.random.randint(0, len(now_folder))
        image = cv2.imread(os.path.join('../../data_real/', now_folder[rand_idx]))
        image_bgr = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        b_image = np.where((image > 127), 0, 255).astype(np.uint8)
        # rand_row = np.random.randint(0, image.shape[0]-1 -256)
        # rand_col = np.random.randint(0, image.shape[1]-1 -256)

        image_resize = cv2.resize(image_bgr, (1200, 900))
        drawer = MouseSquareDrawer(image_resize, square_size=int(256 * 900 / image.shape[0]))
        drawer.run()

        points = drawer.get_clicked_points()
        if len(points) == 0:
            continue
        col, row = points[0]

        row = row * (image.shape[0] / 900)
        col = col * (image.shape[1] / 1200)

        row = int(row)
        col = int(col)

        image_cropped = image_bgr[row:row + 256, col:col+256]
        b_image_cropped = b_image[row:row + 256, col:col+256]

        if Debug:
            extension.image_show(image_cropped)
            extension.image_show(b_image_cropped)
        is_ok = True
        if is_ok:
            cv2.imwrite(f'./mask/{cnt}.png', b_image_cropped)
            cv2.imwrite(f'./data/{cnt}.png', image_cropped)
            cnt += 1
        if start_num + num_target < cnt:
            break

if __name__ == '__main__':
    binary()


