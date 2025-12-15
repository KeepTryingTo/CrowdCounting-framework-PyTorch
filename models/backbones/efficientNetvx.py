"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/18-13:06
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import models


class EfficientNetVxBackbone(nn.Module):
    def __init__(self, pretrained=True, version='b0'):
        super().__init__()
        self.version = version

        version_map = {
            'b0': models.efficientnet_b0,
            'b1': models.efficientnet_b1,
            'b2': models.efficientnet_b2,
            'b3': models.efficientnet_b3,
            'b4': models.efficientnet_b4,
            'b5': models.efficientnet_b5,
            'b6': models.efficientnet_b6,
            'b7': models.efficientnet_b7
        }

        backbone = version_map[version](pretrained=pretrained)
        features = list(backbone.features)

        # 精确划分阶段（基于EfficientNet的标准结构）
        # Stage 0: 初始卷积和BN (1/2分辨率)
        self.stage0 = nn.Sequential(*features[0:2])

        # Stage 1: 第一个MBConv块序列 (1/4分辨率)
        self.stage1 = nn.Sequential(*features[2:3])

        # Stage 2: 第二个MBConv块序列 (1/8分辨率)
        self.stage2 = nn.Sequential(*features[3:5])

        # Stage 3: 第三个MBConv块序列 (1/16分辨率)
        self.stage3 = nn.Sequential(*features[5:7])

        # Stage 4: 第四个MBConv块序列 (1/32分辨率)
        self.stage4 = nn.Sequential(*features[7:9])

        # 通道配置（包含所有阶段的输出通道）
        self.channel_config = {
            'b0': [32, 16, 24, 40, 80, 112, 192, 320, 1280],
            'b1': [32, 16, 24, 40, 80, 112, 192, 320, 1280],
            'b2': [32, 16, 24, 48, 88, 120, 208, 352, 1408],
            'b3': [40, 24, 32, 48, 96, 136, 232, 384, 1536],
            'b4': [48, 24, 32, 56, 112, 160, 272, 448, 1792],
            'b5': [48, 24, 40, 64, 128, 176, 304, 512, 2048],
            'b6': [56, 32, 40, 72, 144, 200, 344, 576, 2304],
            'b7': [64, 32, 48, 80, 160, 224, 384, 640, 2560]
        }

    def forward(self, x: Tensor) -> dict:
        features = {}

        x = self.stage0(x)  # 1/2分辨率
        features['stage0'] = x

        x = self.stage1(x)  # 1/4分辨率
        features['stage1'] = x

        x = self.stage2(x)  # 1/8分辨率
        features['stage2'] = x

        x = self.stage3(x)  # 1/16分辨率
        features['stage3'] = x

        x = self.stage4(x)  # 1/32分辨率
        features['stage4'] = x

        return features
