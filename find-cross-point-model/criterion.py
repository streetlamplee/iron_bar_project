import torch
from torch import nn
import torch.nn.functional as F

class pointFinderCriterion(nn.Module):
    def __init__(self, pos_weight = 5.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.register_buffer('pos_weight_tensor', torch.tensor(pos_weight))

    def forward(self, x, x_logit, target):
        mse_loss = 0.0
        bce_loss = 0.0

        for i in range(4):
            x_idx = i * 3 + 0
            y_idx = i * 3 + 1
            objectness = i * 3 + 2

            mask = target[:, objectness, :, :] > 0

            if mask.any():
                mse_loss += self.mse_loss(x[:, x_idx, :, :][mask], target[:, x_idx, :, :][mask])
                mse_loss += self.mse_loss(x[:, y_idx, :, :][mask], target[:, y_idx, :, :][mask])

            pos_weight = self.pos_weight_tensor.to(target.device)

            bce_loss += F.binary_cross_entropy_with_logits(
                x_logit[:, objectness, :, :],
                target[:, objectness, :, :],
                pos_weight = pos_weight
            )

        total_loss =  5 * mse_loss + bce_loss

        return total_loss, mse_loss, bce_loss