"""
학습된 모델로 사진 한 장에서 철근 영역을 예측한다.

메인 파이프라인이 사진마다 호출하는 모듈이다.
경로가 "src/iron_bar_segmentation/models" 로 고정되어 있어 프로젝트 루트에서 실행해야 한다.
"""

import numpy as np
import torch
import cv2
import torch.nn as nn
from torchvision.models.resnet import resnet101, ResNet101_Weights
import torch.nn.functional as F
from torchvision.transforms import v2
from iron_bar_segmentation.model import DeepLabv3Plus
import etc.extension as extension
import os


def get_model(num_classes):
    model = DeepLabv3Plus(num_classes)
    return model

def predict(input):
    """
    사진 한 장을 넣어 철근 확률맵을 얻는다.

    :param input: RGB 이미지 (H, W, 3). BGR을 넣으면 색 순서가 달라 결과가 나빠진다.
    :return: 0~255 grayscale 확률맵 (255에 가까울수록 철근일 가능성이 높다)

    주의: 호출할 때마다 모델 파일을 다시 읽고 가중치를 올린다.
          사진이 여러 장이면 그만큼 반복되므로 느리다.
    """
    # models 폴더에서 가장 최근에 저장된 checkpoint를 자동으로 고른다.
    model_file = extension.get_latest_pth_file("src/iron_bar_segmentation/models", '.pth')
    # checkpoint = torch.load('/home/user/PycharmProjects/iron_bar_sample_project/iron_bar_segmentation/models/20250616_174956/epoch00231.pth', map_location=torch.device('cpu'))
    checkpoint = torch.load(os.path.join("src/iron_bar_segmentation/models", model_file), map_location=torch.device('cpu'))
    print(os.path.join('./iron_bar_segmentati1on/models', model_file))
    model = get_model(1)
    # 저장된 checkpoint에는 optimizer 등도 함께 있으므로 모델 가중치만 꺼내 넣는다.
    model.load_state_dict(checkpoint['model_state_dict'])
    # 추론 모드로 전환 (BatchNorm/Dropout 동작이 학습 때와 달라진다).
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 학습 때와 똑같은 전처리를 거쳐야 결과가 맞는다.
    # 0~255 -> 0~1 스케일 조정
    test_input = torch.tensor(input, dtype=torch.float32)
    test_input /= 255
    # ImageNet 통계로 정규화 (backbone이 이 기준으로 사전학습되어 있다)
    normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # (H, W, C) -> (C, H, W) 로 축 순서 변경 후, 배치 차원을 붙여 (1, C, H, W) 로 만든다.
    test_input =  normalize(test_input.permute(2,0,1))
    test_input = test_input.unsqueeze(0)
    test_input = test_input.to(device)
    with torch.no_grad():
        output = model(test_input)
        print(output.shape)
        # 모델은 logit을 내놓으므로 sigmoid로 0~1 확률로 바꾼다.
        output = torch.sigmoid(output)
    # 배치 차원을 떼고 (C, H, W) -> (H, W, C) 로 되돌린 뒤 0~255 이미지로 변환한다.
    result = (output.squeeze(0).permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    return result

if __name__ == '__main__':
    # 이 파일만 단독 실행할 때 쓰는 확인용 코드.
    # 경로가 예전 절대경로라 그대로는 동작하지 않으므로, 확인하려면 경로를 바꿔야 한다.
    input = cv2.imread('/home/user/PycharmProjects/iron_bar_sample_project/data_real/0.jpg')
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    output = predict(input)

    output = cv2.resize(output, (1200, 900))
    input = cv2.resize(cv2.cvtColor(input, cv2.COLOR_RGB2BGR), (1200, 900))
    extension.image_show(input)
    extension.image_show(output)
    # 확률맵을 절반(127) 기준으로 잘라 흑백 마스크로 만들어 본다.
    output_thres = np.where((output > 127), 255, 0).astype(np.uint8)
    extension.image_show(output_thres)