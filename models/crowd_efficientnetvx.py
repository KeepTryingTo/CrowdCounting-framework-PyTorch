"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/18-13:14
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from models.backbones.efficientNetvx import EfficientNetVxBackbone
from models.utils.modules import upsample_bilinear
from models.heads.pred_density import PreDensityHead

from models.utils.flops import compute_flops

class CrowdEfficientNetvx(nn.Module):
    def __init__(self, pretrained = True,version = 'b0'):
        super().__init__()
        self.backbone = EfficientNetVxBackbone(pretrained=pretrained,
                                            version=version)
        if version == 'b0':
            self.head = PreDensityHead(in_channels=80)
        elif version == 'b1':
            self.head = PreDensityHead(in_channels=80)
        elif version == 'b2':
            self.head = PreDensityHead(in_channels=88)
        elif version == 'b3':
            self.head = PreDensityHead(in_channels=96)
        elif version == 'b4':
            self.head = PreDensityHead(in_channels=112)
        elif version == 'b5':
            self.head = PreDensityHead(in_channels=128)
        elif version == 'b6':
            self.head = PreDensityHead(in_channels=144)
        elif version == 'b7':
            self.head = PreDensityHead(in_channels=160)

    def forward(self, x):
        features = self.backbone(x)
        # print('out.size: {}'.format(out.size()))
        out = self.head(features['stage2'])
        out = upsample_bilinear(out, x.size())
        return out


if __name__ == '__main__':
    model = CrowdEfficientNetvx(True, 'b0')
    x = torch.randn(size=(1, 3, 512, 512))
    out = model(x)

    print('out.size: {}'.format(out.size()))
    compute_flops(model, img_size=(512, 512), device = 'cpu')