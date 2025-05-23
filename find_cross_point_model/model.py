import torch
from torch import nn
import timm

class pointFindingModel(nn.Module):
    def __init__(self):
        super().__init__()
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

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=960, out_channels=12, kernel_size=(3, 3), padding=(1, 1)),
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