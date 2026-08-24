"""
학습용 (사진, 마스크) 쌍을 반쯤 수동으로 만드는 도구.

무작위로 고른 사진을 띄우면 사용자가 마우스로 잘라낼 위치를 지정하고,
그 자리에서 256x256 조각과 밝기 기준으로 만든 마스크를 함께 저장한다.

실행: 이 파일이 있는 폴더에서 실행해야 한다 (경로가 상대경로).
      data/ 와 mask/ 에 같은 번호로 짝지어 저장된다.
"""

import numpy as np
import cv2
import os
import extension
from iron_bar_segmentation.data_real.square_drawer import MouseSquareDrawer

# Debug가 True면 저장 전에 잘라낸 조각과 마스크를 창으로 확인한다.
Debug = True

def binary():
    os.makedirs('mask', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    # 원본 사진 폴더에서 이미지 파일만 추린다.
    now_folder = os.listdir('../../data_real/')
    now_folder = [png_file for png_file in now_folder if png_file.endswith('.png') or png_file.endswith('.jpg')]
    # 이미 저장된 파일 중 가장 큰 번호를 찾아 이어서 번호를 매긴다 (기존 데이터를 덮어쓰지 않도록).
    # 주의: data 폴더가 비어 있으면 여기서 에러가 난다.
    start_num = sorted([int(i.replace('.png','')) for i in os.listdir('data')], reverse=True)[0]
    cnt = start_num + 1
    # 이번 실행에서 새로 만들 데이터 개수
    num_target = 20
    while True:
        # 매번 무작위 사진을 골라 다양한 장면이 섞이게 한다.
        rand_idx = np.random.randint(0, len(now_folder))
        image = cv2.imread(os.path.join('../../data_real/', now_folder[rand_idx]))
        image_bgr = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 마스크를 자동으로 만든다: 어두운 픽셀을 철근(255)으로 본다.
        # 밝기만으로 나누는 단순한 방식이라, 조명에 따라 품질이 크게 달라진다.
        b_image = np.where((image > 127), 0, 255).astype(np.uint8)
        # rand_row = np.random.randint(0, image.shape[0]-1 -256)
        # rand_col = np.random.randint(0, image.shape[1]-1 -256)

        # 화면에 맞게 축소해서 보여주고, 잘라낼 위치를 마우스로 고르게 한다.
        # 사각형 크기도 축소 비율에 맞춰 줄여 실제 256x256에 대응시킨다.
        image_resize = cv2.resize(image_bgr, (1200, 900))
        drawer = MouseSquareDrawer(image_resize, square_size=int(256 * 900 / image.shape[0]))
        drawer.run()

        points = drawer.get_clicked_points()
        # 아무 곳도 클릭하지 않고 닫았으면 이 사진은 건너뛴다.
        if len(points) == 0:
            continue
        col, row = points[0]

        # 축소된 화면에서 클릭한 좌표를 원본 해상도 기준으로 되돌린다.
        row = row * (image.shape[0] / 900)
        col = col * (image.shape[1] / 1200)

        row = int(row)
        col = int(col)

        # 클릭 지점을 좌상단으로 삼아 256x256 조각을 잘라낸다 (사진과 마스크 동일 위치).
        image_cropped = image_bgr[row:row + 256, col:col+256]
        b_image_cropped = b_image[row:row + 256, col:col+256]

        if Debug:
            extension.image_show(image_cropped)
            extension.image_show(b_image_cropped)
        # 원래는 사람이 채택 여부를 결정하려던 자리인데, 지금은 항상 저장한다.
        is_ok = True
        if is_ok:
            cv2.imwrite(f'./mask/{cnt}.png', b_image_cropped)
            cv2.imwrite(f'./data/{cnt}.png', image_cropped)
            cnt += 1
        # 목표 개수를 채우면 종료
        if start_num + num_target < cnt:
            break

if __name__ == '__main__':
    binary()


