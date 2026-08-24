"""
철근 segmentation 학습에 쓰는 loss 함수 모음.

사진에서 철근이 차지하는 픽셀은 배경보다 훨씬 적다(class imbalance).
그래서 픽셀 단위로 맞히는 BCE와 겹치는 영역 비율을 보는 Dice를 함께 써서,
"전부 배경"이라고 답해도 loss가 낮아지는 현상을 막는다.
"""

import numpy as np
import torch
import cv2
import torch.nn as nn
from torchvision.models.resnet import resnet101, ResNet101_Weights
import torch.nn.functional as F
from torchvision.transforms import v2

class SegmentationLoss(nn.Module):
    """다중 클래스용 CrossEntropy loss. 현재 학습(클래스 1개)에서는 사용하지 않는다."""
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)



class BCELoss(nn.Module):
    def __init__(self):
        """
        픽셀 하나하나를 철근/배경으로 맞히는 이진 분류 loss.

        (아래 docstring은 Focal Loss를 쓰려던 시점의 설명이 남은 것으로,
         현재 구현은 alpha/gamma 없는 일반 BCE다.)
        Focal Loss를 사용한 Segmentation Loss
        :param alpha: 클래스 가중치(imbalance 조정). 일반적으로 0.25 ~ 1.0 사용.
        :param gamma: 어려운 샘플에 집중하기 위한 조정 값. 일반적으로 2.0 사용.
        """
        super(BCELoss, self).__init__()

    def forward(self, outputs, targets):
        """
        :param outputs: 모델 출력 (logits, shape: [batch_size, num_classes, height, width])
        :param targets: 타겟 레이블 (shape: [batch_size, height, width])
        :return: Focal Loss 값
        """
        # 모델이 sigmoid 없이 logit을 내놓으므로 _with_logits 버전을 쓴다 (수치적으로 더 안정적).
        loss = F.binary_cross_entropy_with_logits(outputs, targets.float())
        return loss

class DiceLoss(nn.Module):
    """
    예측 영역과 정답 영역이 얼마나 겹치는지를 보는 loss.

    철근처럼 대상 픽셀이 적을 때, 배경만 잘 맞혀서 얻는 점수를 배제하고
    "철근 영역 자체를 얼마나 잘 덮었는가"에 집중하게 만든다.
    """
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        # predict = torch.sigmoid(predict)  # (16, 2, 256, 256)
        # predict = torch.clamp(predict, min = 1e-4, max = 1 - 1e-4)
        # (16, 256, 256) → (16, 256, 256, 2) → (16, 2, 256, 256)
        # target_onehot = F.one_hot(target, num_classes=predict.shape[1]).permute(0,3,1,2)  # (N, C, H, W)

        # predict = predict[:,1,:,:]              # >> (16, 256, 256)
        # target = target[:,1,:,:]  # >> (16, 256, 256)
        # 모델 출력은 logit이므로 0~1 확률로 바꾼다.
        predict = predict.sigmoid()

        # 교집합 / 합집합 넓이를 이미지별로 계산 (1e-4는 0으로 나누는 것을 막는 값)
        intersection = (predict * target).sum(dim=(-2, -1))  # (N, C)
        union = predict.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))  # (N, C)

        dice = (2.0 * intersection + 1e-4) / (union + 1e-4)  # (N)

        # dice가 1이면 완전히 겹친 것이므로, loss는 1 - dice 로 정의한다.
        dice_loss = 1 - dice.mean() # >> SCALAR

        return dice_loss