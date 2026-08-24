"""
철근 교차점을 찾는 모델 정의 (MobileNetV3 backbone).

교차점 좌표를 직접 회귀하지 않고, 이미지를 격자로 나눠
칸마다 "점이 있는가 / 칸 안 어디에 있는가"를 예측하는 방식이다.
"""

import torch
from torch import nn
import timm

class pointFindingModel(nn.Module):
    """
    입력 이미지를 격자로 나누고, 격자 칸마다 "여기에 점이 있는지 + 칸 안 어디쯤인지"를 예측한다.
    YOLO의 grid 방식과 같은 구조다.

    출력 채널 3개 = (칸 내 상대 x, 상대 y, 점이 있을 확률).
    """
    def __init__(self):
        super().__init__()
        # 가볍고 빠른 MobileNetV3를 특징 추출기로 쓴다 (사전학습 가중치 사용).
        # 필요한 블록만 꺼내 직접 이어붙여 쓴다.
        model = timm.create_model("mobilenetv3_large_100", pretrained=True)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3]
        self.block4 = model.blocks[4]
        self.block5 = model.blocks[5]
        self.block6 = model.blocks[6]

        # 마지막 feature map을 (rel_x, rel_y, objectness) 3채널로 줄이는 예측 헤드.
        # 이 시점의 feature map 한 칸이 곧 원본 이미지의 한 격자 칸에 대응한다.
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=960, out_channels=480, kernel_size=(3,3), padding=(1,1)),
            nn.Conv2d(in_channels=480, out_channels=3, kernel_size=(3,3), padding=(1,1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.classifier(x)
        return x