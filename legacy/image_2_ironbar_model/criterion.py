import torch
from torch import nn
import torch.nn.functional as F

class lineFinderCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x_logit, target):
        x = torch.sigmoid(x_logit)      # x.shape = (B, 3, 8, 8)

        mse_loss = 0.0
        bce_loss = 0.0
        w_mse = 10
        w_bce = 1
        pos_w = torch.tensor([10/1]).to(self.device)

        point_obj_idx = 0
        x_value_idx = 1
        y_value_idx = 2

        bce_loss += F.binary_cross_entropy_with_logits(x_logit[:,point_obj_idx,:,:],
                                                       target[:,point_obj_idx,:,:],
                                                       pos_weight=pos_w)

        has_point = x[:, point_obj_idx, :, :] >= 0.5

        if torch.any(has_point):
            mse_loss += self.mse_loss(x[:,x_value_idx, :, :][has_point],
                                      target[:,x_value_idx,:,:][has_point])

            mse_loss += self.mse_loss(x[:,y_value_idx,:,:][has_point],
                                      target[:,y_value_idx,:,:][has_point])

        total_loss = w_bce * bce_loss + w_mse * mse_loss

        return total_loss, bce_loss, mse_loss