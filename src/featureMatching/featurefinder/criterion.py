"""
교차점 검출 모델의 loss.

"이 칸에 점이 있는가"(분류)와 "칸 안에서 정확히 어디인가"(좌표 회귀)를 동시에 학습시킨다.
"""

import torch
from torch import nn
import torch.nn.functional as F

class pointFinderCriterion(nn.Module):
    """
    점의 존재 여부(BCE)와 칸 안에서의 위치(MSE)를 함께 학습시키는 loss.
    """
    def __init__(self, pos_weight = 5.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x_logit, target):
        mse_loss = 0.0
        bce_loss = 0.0
        # 위치보다 "점이 있다/없다"를 먼저 맞히는 것이 중요해 BCE 쪽에 5배 가중치를 준다.
        w_mse = 1.0
        w_bce = 5.0

        # 채널이 (x, y, objectness) 3개씩 묶여 있다. 현재는 칸당 점 1개만 사용한다.
        for i in range(1):
            x_idx = i * 3 + 0
            y_idx = i * 3 + 1
            objectness = i * 3 + 2

            bce_loss += F.binary_cross_entropy_with_logits(
                x_logit[:, objectness, :, :],
                target[:, objectness, :, :]
            )

            # 좌표 loss는 실제로 점이 있는 칸에서만 계산한다.
            # 점이 없는 칸의 좌표값은 의미가 없어서, 포함시키면 학습을 방해한다.
            has_point = target[:, objectness, :, :] > 0.5

            if torch.any(has_point):
                mse_loss += self.mse_loss(x_logit[:, x_idx, :, :][has_point].sigmoid(), target[:, x_idx, :, :][has_point])
                mse_loss += self.mse_loss(x_logit[:, y_idx, :, :][has_point].sigmoid(), target[:, y_idx, :, :][has_point])

        total_loss = (w_mse * mse_loss) + (w_bce * bce_loss)

        return total_loss, mse_loss, bce_loss