import numpy as np
import torch
import cv2
import torch.nn as nn
from torchvision.models.resnet import resnet101, ResNet101_Weights
import torch.nn.functional as F
from torchvision.transforms import v2

class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)



class BCELoss(nn.Module):
    def __init__(self):
        """
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
        loss = F.binary_cross_entropy_with_logits(outputs, targets.float())
        return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        # predict = torch.sigmoid(predict)  # (16, 2, 256, 256)
        # predict = torch.clamp(predict, min = 1e-4, max = 1 - 1e-4)
        # (16, 256, 256) → (16, 256, 256, 2) → (16, 2, 256, 256)
        # target_onehot = F.one_hot(target, num_classes=predict.shape[1]).permute(0,3,1,2)  # (N, C, H, W)

        # predict = predict[:,1,:,:]              # >> (16, 256, 256)
        # target = target[:,1,:,:]  # >> (16, 256, 256)
        predict = predict.sigmoid()

        intersection = (predict * target).sum(dim=(-2, -1))  # (N, C)
        union = predict.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))  # (N, C)

        dice = (2.0 * intersection + 1e-4) / (union + 1e-4)  # (N)

        dice_loss = 1 - dice.mean() # >> SCALAR

        return dice_loss