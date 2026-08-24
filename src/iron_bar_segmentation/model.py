"""
철근 segmentation 모델 정의 (DeepLabv3+, ResNet101 backbone).

철근은 사진 전체에 걸쳐 가늘고 길게 뻗어 있어서, 좁은 영역만 보면 배경과 구분하기 어렵다.
DeepLabv3+는 넓은 범위를 한 번에 보는 ASPP와 세밀한 윤곽을 살리는 decoder를 함께 써서
이런 형태를 잡아내는 데 적합하다.
"""

import numpy as np
import torch
import cv2
import torch.nn as nn
from torchvision.models.resnet import resnet101, ResNet101_Weights
import torch.nn.functional as F
from torchvision.transforms import v2

class ASPPPooling_(nn.Sequential):
    """
    이미지 전체를 한 픽셀로 압축했다가 되돌리는 가지.
    "이 사진 전체가 어떤 장면인가"라는 전역 맥락을 ASPP에 더해준다.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1x1로 줄였던 결과를 원래 feature map 크기로 다시 늘려 다른 가지와 합칠 수 있게 한다.
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP_(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    같은 feature map을 서로 다른 간격(dilation)의 convolution으로 훑어,
    좁게 보는 시야와 넓게 보는 시야의 정보를 한꺼번에 모은다.
    가늘지만 길게 이어지는 철근을 놓치지 않기 위한 핵심 부분이다.

    :param atrous_rate: 사용할 dilation 간격 목록. 클수록 더 넓은 범위를 본다.
    """
    def __init__(self, in_channels, out_channels, atrous_rate, dropout_rate=0.1):
        super(ASPP_, self).__init__()
        modules = []
        # 가지 1: 1x1 conv (해당 위치의 정보만 그대로 본다)
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias= False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        )

        # 가지 2~n: dilation을 키운 3x3 conv. rate가 클수록 더 멀리 떨어진 픽셀까지 함께 본다.
        rates = tuple(atrous_rate)
        for rate in rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )

        # 마지막 가지: 전역 평균 (사진 전체의 맥락)
        modules.append(ASPPPooling_(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

        # 모든 가지의 출력을 이어붙인 뒤 다시 out_channels 로 압축한다.
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 모든 가지에 같은 입력을 통과시키고 채널 방향으로 이어붙인다.
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

        # (아래 줄은 return 뒤라 실행되지 않는 잔재다)
        self

class DeepLabv3Plus(nn.Module):
    """
    :param num_classes: 출력 채널 수. 이 프로젝트는 "철근인가 아닌가" 하나만 보므로 1이며,
                        출력은 logit이라 사용 시 sigmoid를 따로 적용해야 한다.
    """
    def __init__(self, num_classes):
        super(DeepLabv3Plus, self).__init__()

        # ImageNet으로 미리 학습된 ResNet101을 특징 추출기로 사용한다 (학습 데이터가 적어 전이학습이 유리).
        # replace_stride_with_dilation의 마지막 True: 마지막 단계에서 해상도를 더 줄이는 대신
        # dilation을 써서, 가는 철근이 뭉개지지 않게 한다.
        # ResNet backbone
        resnet = resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V2,
            progress=False,
            replace_stride_with_dilation=[False, False, True]
        )
        # self.layer0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(15,15), padding=(7,7), stride=(2,2), bias=False), resnet.bn1, resnet.relu, resnet.maxpool)  # Initial layers
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # Initial layers
        self.layer1 = nn.Sequential(resnet.layer1)  # Low-level features
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)  # High-level features

        # ASPP module
        # 2048채널의 고수준 특징을 256채널로 줄이면서 여러 시야의 정보를 합친다.
        self.aspp = ASPP_(2048, 256, [6, 12, 24, 36])

        # 초반 layer1의 특징(윤곽 같은 세밀한 정보)을 48채널로 줄여 decoder에서 다시 합친다.
        # Low-level features convolution
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder
        # 입력 304채널 = ASPP 출력 256 + low-level 48
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        # 마지막에 원본 해상도로 되돌리기 위해 입력 크기를 기억해둔다.
        input_size = x.size()[2:]

        # Encoder: 해상도를 줄여가며 점점 추상적인 특징을 뽑는다.
        x = self.layer0(x)
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Decoder: ASPP 결과를 초반 특징의 해상도까지 키운 뒤 둘을 합친다.
        # "무엇인지"는 깊은 층이, "경계가 어디인지"는 얕은 층이 담당하는 구조다.
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)

        low_level_feat = self.low_level_conv(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)

        # 마지막으로 입력 사진과 같은 크기로 복원한다.
        # 주의: sigmoid를 적용하지 않은 logit을 반환한다 (loss 계산 편의를 위함).
        # Final upsampling
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        # x = torch.sigmoid(x)
        return x