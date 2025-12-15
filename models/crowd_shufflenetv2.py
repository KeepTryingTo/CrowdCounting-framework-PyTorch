"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/18-10:33
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from models.backbones.shufflenetv1 import ShuffleNetV2Backbone
from models.utils.modules import upsample_bilinear
from models.heads.pred_density import PreDensityHead

from models.utils.flops import compute_flops

class CrowdShuffleNetv2(nn.Module):
    def __init__(self, pretrained = True,version = 'x0_5'):
        super().__init__()
        self.backbone = ShuffleNetV2Backbone(pretrained=pretrained,
                                            version=version)
        if version == 'x0_5':
            self.head = PreDensityHead(in_channels=96)
        elif version == 'x1_0':
            self.head = PreDensityHead(in_channels=232)
        elif version == 'x1_5':
            self.head = PreDensityHead(in_channels=352)
        elif version == 'x2_0':
            self.head = PreDensityHead(in_channels=488)

    def forward(self, x):
        features = self.backbone(x)
        # print('out.size: {}'.format(out.size()))
        out = self.head(features['stage3'])
        out = upsample_bilinear(out, x.size())
        return out


if __name__ == '__main__':
    model = CrowdShuffleNetv2(True, 'x2_0')
    x = torch.randn(size=(1, 3, 512, 512))
    out = model(x)

    print('out.size: {}'.format(out.size()))
    compute_flops(model, img_size=(512, 512), device = 'cpu')