"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/18-10:26
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


class ShuffleNetV2Backbone(nn.Module):
    def __init__(self, pretrained=True, version='x1_0'):
        super().__init__()

        self.pretrained = pretrained
        self.version = version

        # 加载预训练模型
        version_map = {
            'x0_5': models.shufflenet_v2_x0_5,
            'x1_0': models.shufflenet_v2_x1_0,
            'x1_5': models.shufflenet_v2_x1_5,
            'x2_0': models.shufflenet_v2_x2_0
        }
        if version == 'x1_5' or version == 'x2_0':
            # 这是因为x1_5和x2_0版本官方没有提供预训练模型
            backbone = version_map[version](pretrained=False)
        else:
            backbone = version_map[version](pretrained=pretrained)

        # 分解ShuffleNetV2为不同阶段
        # Stage1: 初始卷积层
        self.stage1 = nn.Sequential(
            backbone.conv1,
            backbone.maxpool
        )

        # Stage2-4: 主要的阶段块
        self.stage2 = backbone.stage2  # 1/4分辨率
        self.stage3 = backbone.stage3  # 1/8分辨率
        self.stage4 = backbone.stage4  # 1/16分辨率

    def forward(self, x: Tensor) -> dict:
        """返回所有中间层特征"""
        features = {}

        features['input'] = x
        features['stage1'] = self.stage1(x)  # 1/4分辨率
        features['stage2'] = self.stage2(features['stage1'])  # 1/8分辨率
        features['stage3'] = self.stage3(features['stage2'])  # 1/16分辨率
        features['stage4'] = self.stage4(features['stage3'])  # 1/32分辨率

        return features