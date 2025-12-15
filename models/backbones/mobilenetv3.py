"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-21:11
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import models
from models.utils.modules import ConvNormActivation
from models.utils.modules import _make_divisible
from typing import Callable, Any, Optional, List


class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=True, is_mobilenetv3_large=False):
        super().__init__()

        if is_mobilenetv3_large:
            backbone = models.mobilenet_v3_large(pretrained=pretrained)
            # 大型模型的分层
            self.stage1 = nn.Sequential(*list(backbone.features)[:3])  # 112x112
            self.stage2 = nn.Sequential(*list(backbone.features)[3:7])  # 56x56
            self.stage3 = nn.Sequential(*list(backbone.features)[7:13])  # 28x28
            self.stage4 = nn.Sequential(*list(backbone.features)[13:16])  # 14x14
            self.stage5 = nn.Sequential(*list(backbone.features)[16:])  # 7x7
        else:
            backbone = models.mobilenet_v3_small(pretrained=pretrained)
            # 小型模型的分层
            self.stage1 = nn.Sequential(*list(backbone.features)[:2])  # 112x112
            self.stage2 = nn.Sequential(*list(backbone.features)[2:4])  # 56x56
            self.stage3 = nn.Sequential(*list(backbone.features)[4:9])  # 28x28
            self.stage4 = nn.Sequential(*list(backbone.features)[9:])  # 14x14

    def forward(self, x: Tensor) -> dict:
        features = {}

        features['stage1'] = self.stage1(x)  # 高分辨率特征
        features['stage2'] = self.stage2(features['stage1'])  # 中等分辨率
        features['stage3'] = self.stage3(features['stage2'])  # 低分辨率
        features['stage4'] = self.stage4(features['stage3'])  # 最终特征

        if hasattr(self, 'stage5'):
            features['stage5'] = self.stage5(features['stage4'])

        return features

