"""
교차점 학습 데이터를 손으로 만드는 라벨링 도구.

사진을 256x256으로 무작위로 잘라 화면에 띄우면,
사용자가 철근 교차점을 클릭해 정답 좌표를 찍는다.
결과는 data.json 에 (이미지 경로, 좌표 목록) 형태로 쌓인다.
"""

import numpy as np
import cv2
import os
import json

class clickHandler:
    """마우스 클릭 위치를 모아두는 작은 도우미."""
    def __init__(self):
        self.clicked_points = []
    def handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append([x,y])

def make_data(path:str, output_folder:str, n:int):
    """
    사진을 무작위로 잘라 보여주고, 사용자가 교차점을 클릭해 라벨을 만든다.

    조작: 보이는 교차점을 모두 좌클릭 -> ESC를 누르면 저장하고 다음 조각으로 넘어간다.

    :param path: 원본 이미지 폴더
    :param output_folder: 잘라낸 조각과 data.json 을 저장할 폴더
    :param n: 이번 실행에서 새로 만들 조각 수

    결과: output_folder/image/{번호}.png 와, 좌표를 담은 output_folder/data.json
    """
    if not os.path.exists(path):
        raise RuntimeError(f'No Directory Found : {path}')

    # 기존 data.json 이 있으면 이어서 번호를 매기고, 없으면 새로 만든다.
    if os.path.exists(os.path.join(output_folder, 'data.json')):


        crop_height = 256
        crop_width = 256
        with open(os.path.join(output_folder, 'data.json'), 'r') as f:
            json_s = json.load(f)
        cnt = len(json_s.keys()) -1
        num = cnt

    if not os.path.exists(os.path.join(output_folder, 'data.json')):
        os.makedirs(f'{output_folder}', exist_ok=True)
        os.makedirs(f'{output_folder}/image', exist_ok=True)

        crop_height = 256
        crop_width = 256
        json_s = {}
        json_s['len'] = 0
        cnt = 0
        num = 0

    image_filename_list = os.listdir(path)

    while True:
        # random_idx = np.random.randint(0, len(image_filename_list))
        idx = -1
        idx += 1
        image_filename = image_filename_list[idx]
        if not image_filename.endswith(('.png','.jpg')):
            continue

        json_value = {}

        image = cv2.imread(os.path.join(path, image_filename))

        h, w, _ = image.shape

        # 사진에서 256x256 영역을 무작위 위치로 잘라낸다.
        crop_h = np.random.randint(0, h - crop_height+1)
        crop_w = np.random.randint(0, w - crop_width+1)

        image_crop = image[crop_h:crop_h + crop_height, crop_w:crop_w + crop_width]

        # 잘라낸 조각을 띄우고 교차점을 클릭받는다. 클릭할 때마다 빨간 점으로 표시된다.
        click = clickHandler()
        image_crop_copy = image_crop.copy()
        cv2.imshow("Please click the Crossed point", image_crop_copy)
        cv2.setMouseCallback("Please click the Crossed point", click.handler)
        while True:
            key = cv2.waitKey(1)
            if len(click.clicked_points) != 0:
                cv2.circle(image_crop_copy, click.clicked_points[-1], radius=3, thickness=-1, color = (0,0,255))
            cv2.imshow("Please click the Crossed point", image_crop_copy)

            # ESC를 누르면 이 조각의 라벨링을 마친다.
            if key == 27:
                break

        json_value['filename'] = os.path.join(f'{output_folder}/image', f"{cnt}.png")

        cv2.imwrite(os.path.join(f'{output_folder}/image', f"{cnt}.png"), image_crop)

        # 클릭한 좌표들이 이 조각의 정답이 된다.
        json_value['mask'] = click.clicked_points
        json_s['len'] += len(click.clicked_points)

        json_s[f'{cnt}'] = json_value
        cnt += 1

        # 목표한 개수를 채우면 종료
        if len(json_s) - num >= n+1:
            break

    with open(os.path.join(output_folder, 'data.json'), 'w') as json_file:
        json.dump(json_s, json_file, indent=1)


if __name__ == "__main__":
    make_data('../warp_image', 'data', 5)