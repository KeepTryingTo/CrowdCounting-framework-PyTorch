"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-20:54
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class MobileNetV1Backbone(nn.Module):
    def __init__(self, num_classes = 1000):
        super().__init__()

        self.model = nn.Sequential(
            self.conv_bn(3, 32, 2),
            self.conv_dw(32, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def conv_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def conv_dw(self, inp, oup, stride):
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x