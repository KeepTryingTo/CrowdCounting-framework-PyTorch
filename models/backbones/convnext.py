"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/18-11:00
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import models


class ConvNeXtBackbone(nn.Module):
    def __init__(self, pretrained=True, version='base'):
        super().__init__()
        self.version = version

        version_map = {
            'tiny': models.convnext_tiny,
            'small': models.convnext_small,
            'base': models.convnext_base,
            'large': models.convnext_large
        }

        backbone = version_map[version](pretrained=pretrained)

        # ConvNeXt的features属性包含所有层
        all_layers = list(backbone.features)

        # 精确划分阶段（基于ConvNeXt的标准结构）
        # Stage 0: Stem (初始卷积和下采样)
        self.stage0 = nn.Sequential(*all_layers[0:2])

        # Stage 1: 第一个下采样 + 块序列
        self.stage1 = nn.Sequential(*all_layers[2:4])

        # Stage 2: 第二个下采样 + 块序列
        self.stage2 = nn.Sequential(*all_layers[4:6])

        # Stage 3: 第三个下采样 + 块序列
        self.stage3 = nn.Sequential(*all_layers[6:8])

        # Stage 4: 最后的块序列
        self.stage4 = nn.Sequential(*all_layers[8:])

        # 通道配置
        self.channel_config = {
            'tiny': [96, 192, 384, 768, 768],
            'small': [96, 192, 384, 768, 768],
            'base': [128, 256, 512, 1024, 1024],
            'large': [192, 384, 768, 1536, 1536]
        }

    def forward(self, x: Tensor) -> dict:
        features = {}

        x = self.stage0(x)  # 1/4分辨率
        features['stage0'] = x

        x = self.stage1(x)  # 1/8分辨率
        features['stage1'] = x

        x = self.stage2(x)  # 1/16分辨率
        features['stage2'] = x

        x = self.stage3(x)  # 1/32分辨率
        features['stage3'] = x

        x = self.stage4(x)  # 1/32分辨率（通道增加）
        features['stage4'] = x

        return features
