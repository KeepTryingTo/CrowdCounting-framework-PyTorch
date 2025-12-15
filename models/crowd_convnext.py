"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/18-11:08
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from models.backbones.convnext import ConvNeXtBackbone
from models.utils.modules import upsample_bilinear
from models.heads.pred_density import PreDensityHead

from models.utils.flops import compute_flops

class CrowdMobileNetv3(nn.Module):
    def __init__(self, pretrained = True, version = 'base'):
        super().__init__()
        self.backbone = ConvNeXtBackbone(pretrained=pretrained,
                                        version=version)
        if version == 'base':
            self.head = PreDensityHead(in_channels=512)
        elif version == 'tiny':
            self.head = PreDensityHead(in_channels=384)
        elif version == 'small':
            self.head = PreDensityHead(in_channels=384)
        elif version == 'large':
            self.head = PreDensityHead(in_channels=768)

    def forward(self, x):
        features = self.backbone(x)
        # print('out.size: {}'.format(out.size()))
        out = self.head(features['stage2'])
        out = upsample_bilinear(out, x.size())
        return out


if __name__ == '__main__':
    model = CrowdMobileNetv3(True, 'large')
    x = torch.randn(size=(1, 3, 512, 512))
    out = model(x)

    print('out.size: {}'.format(out.size()))
    compute_flops(model, img_size=(512, 512), device = 'cpu')