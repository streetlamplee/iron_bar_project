"""
이미 만들어둔 라벨 데이터에 새 데이터를 추가하는 도구.

data_processing.make_data 와 목적은 같지만, 이쪽은 기존 data.json 에 이어붙이는 용도다.
사진을 512x512로 잘라 보여주고, 쓸 만한 조각일 때만 사용자가 교차점을 클릭해 추가한다.
"""

import os
import json
import numpy as np
import cv2
import extension
import torch
from model import pointFindingModel
from torchvision import transforms
from nms import nms

def predict(input_image:np.ndarray, m = None):
    """
    현재 모델로 교차점을 예측해 원본 위에 찍어 돌려준다.
    라벨링을 돕기 위한 참고용이며, 아래 make_more_data 에서는 현재 호출하지 않는다.

    :param m: 사용할 checkpoint 경로. None이면 models 폴더의 최신 파일을 쓴다.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if m is None:
        model_folder = './models'
        model_filename = os.path.join(model_folder, extension.get_latest_pth_file(model_folder, '.pth'))
        checkpoint = torch.load(model_filename)
        model = pointFindingModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
    else:
        model = pointFindingModel()
        checkpoint = torch.load(m)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
    # 학습 때와 같은 전처리 (RGB -> 0~1 -> ImageNet 정규화 -> (1, C, H, W))
    t_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    t_image = torch.tensor(t_image, dtype = torch.float32)
    t_image /= 255
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    t_image = t_image.permute(2,0,1).unsqueeze(0)
    t_image = normalize(t_image)
    t_image = t_image.to(device)
    with torch.no_grad():
        output = model(t_image)
        print(output.shape)
        output = torch.sigmoid(output)
        # 중복 예측 제거
        keypoints = nms(output, 7)
    res = input_image.copy()
    for keypoint in keypoints:
        h, w, o = keypoint
        h = int(h)
        w = int(w)
        if o >= .5:
            cv2.circle(res, (h,w), 2, (0,0,255), -1)
    os.makedirs('result', exist_ok=True)
    # cv2.imwrite(os.path.join('result', test_image), res)
    # extension.image_show(res)
    return res, input_image

class clickHandler:
    """마우스 클릭 위치를 모아두는 도우미. (data_processing.py 에도 같은 클래스가 있다.)"""
    def __init__(self):
        self.clicked_points = []
    def handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append([x,y])

def clicked_point_finder(image:np.ndarray):
    """
    이미지를 띄우고 교차점을 클릭받는다.
    조작: 좌클릭으로 점 찍기 -> Enter로 확정.
    :return: 클릭된 좌표 목록
    """
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

def make_more_data(num:int, data_path:str, origin_data_json_path:str):
    """
    기존 data.json 에 새 라벨 데이터를 num 개 추가한다.

    진행 방식: 무작위로 자른 조각을 보여준 뒤,
              터미널에 1을 입력하면 라벨링을 진행하고 0을 입력하면 건너뛴다.

    :param num: 추가할 데이터 개수
    :param data_path: 원본 사진 폴더
    :param origin_data_json_path: 이어붙일 data.json 경로
    """
    data_list = os.listdir(data_path)
    with open(origin_data_json_path, 'r') as f:
        data_json = json.load(f)
    # 기존 개수를 기억해두고, 그만큼 늘어나면 종료한다.
    # data.json 의 "len" 항목은 데이터가 아니라 점 개수 합계이므로 번호 계산에서 뺀다.
    before_data_len = len(data_json)
    if 'len' in data_json.keys():
        cnt = before_data_len - 1
    else:
        cnt = before_data_len
    while True:
        data_file_name_iter = os.path.join(data_path, data_list[np.random.randint(0, len(data_list))])
        image_iter = cv2.imread(data_file_name_iter)
        # 무작위 사진에서 무작위 위치의 512x512 영역을 잘라낸다.
        random_x = np.random.randint(0, image_iter.shape[0] - 512 + 1)
        random_y = np.random.randint(0, image_iter.shape[1] - 512 + 1)
        image = image_iter[random_x: random_x + 512, random_y: random_y + 512]
        # res_iter, image = predict(image_iter)
        # extension.image_show(res_iter)
        extension.image_show(image)
        # 창에 뜬 조각을 보고 터미널에 1(사용) / 0(건너뛰기)을 입력한다.
        # 철근이 잘 보이지 않는 조각을 걸러내기 위한 단계다.
        if int(input()):
            clicked_point = clicked_point_finder(image)
            sub_dict = {
                'filename': f'data/image/{cnt}.png',
                'mask': clicked_point
            }
            if 'len' in data_json.keys():
                data_json['len'] += len(clicked_point)
            data_json[f'{cnt}'] = sub_dict



            # 조각 이미지를 저장하고, 매번 json도 함께 갱신한다 (중간에 멈춰도 결과가 남도록).
            cv2.imwrite(origin_data_json_path.replace('data.json', f'image/{cnt}.png'), image)
            with open(origin_data_json_path, 'w') as f:
                json.dump(data_json, f, indent=2)
            cnt += 1


        if len(data_json) >= before_data_len + num:
            break

    return

if __name__ == "__main__":
    extension.set_seed(56)
    make_more_data(50, './data/origin', './data/data.json')
