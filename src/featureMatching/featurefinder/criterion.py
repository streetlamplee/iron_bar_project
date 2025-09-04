import torch
from torch import nn
import torch.nn.functional as F

class pointFinderCriterion(nn.Module):
    def __init__(self, pos_weight = 5.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x_logit, target):
        mse_loss = 0.0
        bce_loss = 0.0
        w_mse = 1.0
        w_bce = 5.0

        for i in range(1):
            x_idx = i * 3 + 0
            y_idx = i * 3 + 1
            objectness = i * 3 + 2

            bce_loss += F.binary_cross_entropy_with_logits(
                x_logit[:, objectness, :, :],
                target[:, objectness, :, :]
            )

            has_point = target[:, objectness, :, :] > 0.5

            if torch.any(has_point):
                mse_loss += self.mse_loss(x_logit[:, x_idx, :, :][has_point].sigmoid(), target[:, x_idx, :, :][has_point])
                mse_loss += self.mse_loss(x_logit[:, y_idx, :, :][has_point].sigmoid(), target[:, y_idx, :, :][has_point])

        total_loss = (w_mse * mse_loss) + (w_bce * bce_loss)

        return total_loss, mse_loss, bce_loss