import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
# from timm import create_model
from pprint import pprint
from torchvision.models.resnet import resnet101, ResNet101_Weights

class ASPPPooling_(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP_(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, dropout_rate=0.1):
        super(ASPP_, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias= False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        )

        rates = tuple(atrous_rate)
        for rate in rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )

        modules.append(ASPPPooling_(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

        self

class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabv3Plus, self).__init__()

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
        self.aspp = ASPP_(2048, 256, [6, 12, 24, 36])

        # Low-level features convolution
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.size()[2:]

        # Encoder
        x = self.layer0(x)
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)

        low_level_feat = self.low_level_conv(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)

        # Final upsampling
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        # x = torch.sigmoid(x)
        return x

def convert_bn_to_gn(module):
    """
    모델 내 모든 BatchNorm2d -> GroupNorm 변환 (32 groups 기본)
    """
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = nn.GroupNorm(num_groups=min(32, module.num_features // 4), num_channels=module.num_features)
    for name, child in module.named_children():
        module_output.add_module(name, convert_bn_to_gn(child))
    return module_output

def convert_bn_to_in(module):
    """
    모델 내 모든 BatchNorm2d -> InstanceNorm2d 변환
    """
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = nn.InstanceNorm2d(
            num_features=module.num_features,
            affine=True,  # BatchNorm처럼 scale/shift 파라미터 사용하려면 True
            track_running_stats=False  # InstanceNorm은 보통 running stats 사용 안 함
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_bn_to_in(child))
    return module_output

# 학습을 위한 손실 함수
class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.criterion(outputs, targets)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Focal Loss를 사용한 Segmentation Loss
        :param alpha: 클래스 가중치(imbalance 조정). 일반적으로 0.25 ~ 1.0 사용.
        :param gamma: 어려운 샘플에 집중하기 위한 조정 값. 일반적으로 2.0 사용.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 클래스 불균형에 따른 가중치
        self.gamma = gamma  # 어려운 샘플에 대한 가중치

    def forward(self, outputs, targets):
        """
        :param outputs: 모델 출력 (logits, shape: [batch_size, num_classes, height, width])
        :param targets: 타겟 레이블 (shape: [batch_size, height, width])
        :return: Focal Loss 값
        """
        # 소프트맥스 확률로 변환
        probs = F.softmax(outputs, dim=1)

        # 타겟의 원-핫 인코딩 (CrossEntropyLoss와 호환되는 타겟)
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float() # (N, C, H, W)

        # 타겟 픽셀의 확률만 추출
        probs_target = (probs * targets_one_hot).sum(dim=1)  # Shape: [N, H, W] : 현재 batch 내에서, N 번째 사진의 (H, W) 픽셀이 배경이 아닐 확률

        # Focal Loss 계산
        focal_weight = self.alpha * (1 - probs_target) ** self.gamma  # 가중치 적용, (1 - probs_target) : 배경일 확률, (N, H, W)
        loss = -focal_weight * torch.log(probs_target + 1e-6)  # 로그 손실

        return loss.mean()  # 평균 손실 반환

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
        predict = predict.squeeze(1)
        intersection = (predict * target).sum(dim=(-2, -1))  # (N, C)
        union = predict.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))  # (N, C)

        dice = (2.0 * intersection + 1e-4) / (union + 1e-4)  # (N)

        dice_loss = 1 - dice.mean() # >> SCALAR

        return dice_loss

# class Head(nn.Module):
#     def __init__(self,
#                  high_level_channels=2048,
#                  low_level_channels=256,
#                  num_classes=1,
#                  project_channels=48,
#                  aspp_mid_channels=256,
#                  aspp_last_channels=256,
#                  classifier_channels=256,
#                  aspp_dilates=None):
#         super(Head, self).__init__()
#
#         if aspp_dilates is None:
#             aspp_dilates = [6, 12, 18]
#
#         self.project = nn.Sequential(
#             nn.Conv2d(low_level_channels, project_channels, 1, bias=False),
#             nn.BatchNorm2d(project_channels),
#             nn.ReLU(inplace=True),
#         )
#
#         self.aspp = ASPP(high_level_channels, aspp_mid_channels, aspp_last_channels, aspp_dilates)
#
#         self.classifier = nn.Sequential(
#             nn.Conv2d(project_channels + aspp_last_channels, classifier_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(classifier_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(classifier_channels, num_classes, 1)
#         )
#
#         for m in [self.project, self.aspp, self.classifier]:
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, lf, hf):
#         lf = self.project(lf)
#         hf = self.aspp(hf)
#         hf = F.interpolate(hf, scale_factor=4, mode='bilinear', align_corners=False)
#         cf = torch.cat([lf, hf], dim=1)
#         result = self.classifier(cf)
#         return F.interpolate(result, scale_factor=4, mode='bilinear', align_corners=False)
#
# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         super(ASPPConv, self).__init__(
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#
# class ASPPPooling(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__()
#         self.pooling = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         size = x.shape[-2:]
#         x = self.pooling(x)
#         x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
#         return x
#
#
# class ASPP(nn.Module):
#     def __init__(self,
#                  in_channels=2048,
#                  out_channels=256,
#                  final_out_channels=256,
#                  atrous_rates=None):
#
#         super(ASPP, self).__init__()
#
#         if atrous_rates is None:
#             atrous_rates = [6, 12, 18]
#
#         # 1x1 Conv
#         modules = [nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )]
#
#         # 3x3 Dilated Convs
#         for rate in atrous_rates:
#             modules.append(ASPPConv(in_channels, out_channels, rate))
#
#         # ASPP Pooling
#         modules.append(ASPPPooling(in_channels, out_channels))
#         self.convs = nn.ModuleList(modules)
#
#         self.project = nn.Sequential(
#             nn.Conv2d(len(modules) * out_channels, final_out_channels, 1, bias=False),
#             nn.BatchNorm2d(final_out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1))
#
#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)
#
# class SegmentationModel(nn.Module):
#     def __init__(self,
#                  project_channels=48,
#                  aspp_mid_channels=256,
#                  aspp_last_channels=256,
#                  classifier_channels=256,
#                  aspp_dilates: List[int] | None = None
#                  ):
#         super().__init__()
#         if aspp_dilates is None:
#             aspp_dilates = [6, 12, 18]
#
#         m = resnet101(
#             weights=ResNet101_Weights.IMAGENET1K_V2,
#             progress=False,
#             replace_stride_with_dilation=[False, False, True]
#         )
#         hf_chan, lf_chan = 2048, 256
#
#         self.lf = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
#         self.hf = nn.Sequential(m.layer2, m.layer3, m.layer4)
#         self.head = Head(
#             high_level_channels=hf_chan, low_level_channels=lf_chan, num_classes=1,
#             project_channels=project_channels, aspp_mid_channels=aspp_mid_channels,
#             aspp_last_channels=aspp_last_channels, classifier_channels=classifier_channels,
#             aspp_dilates=aspp_dilates
#         )
#
#     def forward(self, x: torch.tensor):
#         lf = self.lf(x)
#         hf = self.hf(lf)
#         return self.head(lf, hf)

def get_model(num_classes, norm='bn'):
    model = DeepLabv3Plus(num_classes)
    # model = SegmentationModel()
    if norm == "gn":
        model = convert_bn_to_gn(model)
    elif norm == "in":
        model = convert_bn_to_in(model)
    else:
        pass
    return model