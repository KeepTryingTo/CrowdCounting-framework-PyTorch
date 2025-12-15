"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/8/7-19:21
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from models.utils.modules import upsample_bilinear

class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(Backbone, self).__init__()

        # TODO frontend feature exactor
        model = list(models.vgg16(pretrained=pretrained).features.children())
        self.feblock1 = nn.Sequential(*model[:16])
        self.feblock2 = nn.Sequential(*model[16:23])
        self.feblock3 = nn.Sequential(*model[23:30])

        # backend
        self.beblock3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.beblock2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.beblock1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pred_den = nn.Sequential(
            nn.Conv2d(in_channels=896, out_channels=1, kernel_size=1)
        )

    def forward(self, x):
        b,c,h,w = x.size()
        x = self.feblock1(x)
        x1 = x
        x = self.feblock2(x)
        x2 = x
        x = self.feblock3(x)

        # print('x1.shape: {}'.format(x1.size()))
        # print('x2.shape: {}'.format(x2.size()))
        # print('x3.shape: {}'.format(x.size()))

        # decoding stage
        x = self.beblock3(x)
        x3_ = x
        x = upsample_bilinear(x, x2.shape)
        x = torch.cat([x, x2], 1)

        x = self.beblock2(x)
        x2_ = x
        x = upsample_bilinear(x, x1.shape)
        x = torch.cat([x, x1], 1)

        x1_ = self.beblock1(x)

        x2_ = upsample_bilinear(x2_, x1.shape)
        x3_ = upsample_bilinear(x3_, x1.shape)

        x = torch.cat([x1_, x2_, x3_], 1)

        return x

def demo():
    x = torch.zeros(size=(1,3,320,320))
    model = Backbone()
    out = model(x)
    print('out.shape: {}'.format(out.size()))

if __name__ == '__main__':
    demo()
    pass