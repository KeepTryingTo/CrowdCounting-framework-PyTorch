"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-21:08
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from models.backbones.mobilenetv2 import MobileNetV2Backbone
from models.utils.modules import upsample_bilinear
from models.heads.pred_density import PreDensityHead

from models.utils.flops import compute_flops

class CrowdMobileNetv2(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        self.backbone = MobileNetV2Backbone(pretrained=pretrained)
        self.head = PreDensityHead(in_channels=1280)

    def forward(self, x):
        out = self.backbone(x)
        # print('out.size: {}'.format(out.size()))
        out = self.head(out)
        out = upsample_bilinear(out, x.size())
        return out


if __name__ == '__main__':
    model = CrowdMobileNetv2(True)
    x = torch.randn(size=(1, 3, 512, 512))
    out = model(x)

    print('out.size: {}'.format(out.size()))
    compute_flops(model, img_size=(512, 512), device = 'cpu')