"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/18-10:18
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from models.backbones.mobilenetv3 import MobileNetV3Backbone
from models.utils.modules import upsample_bilinear
from models.heads.pred_density import PreDensityHead

from models.utils.flops import compute_flops

class CrowdMobileNetv3(nn.Module):
    def __init__(self, pretrained = True,is_mobilenetv3_large = True):
        super().__init__()
        self.backbone = MobileNetV3Backbone(pretrained=pretrained,
                                            is_mobilenetv3_large=is_mobilenetv3_large)
        if is_mobilenetv3_large:
            self.head = PreDensityHead(in_channels=112)
        else:
            self.head = PreDensityHead(in_channels=48)

    def forward(self, x):
        features = self.backbone(x)
        # print('out.size: {}'.format(out.size()))
        out = self.head(features['stage3'])
        out = upsample_bilinear(out, x.size())
        return out


if __name__ == '__main__':
    model = CrowdMobileNetv3(True, False)
    x = torch.randn(size=(1, 3, 512, 512))
    out = model(x)

    print('out.size: {}'.format(out.size()))
    compute_flops(model, img_size=(512, 512), device = 'cpu')