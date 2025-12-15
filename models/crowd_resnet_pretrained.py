"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/4/5-13:10
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn
from torchvision import models
import thop
import torch.nn.functional as F
from models.heads.pred_density import PreDensityHead
from models.utils.flops import compute_flops
from models.utils.modules import upsample_bilinear

class ResNet50Crowd(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(pretrained=True,progress=True)
        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.head = PreDensityHead(in_channels=1024)

    def forward(self,x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        final_out = self.layer4(out)

        out = self.head(out)
        out = upsample_bilinear(out,x.size())

        return out


if __name__ == '__main__':
    model = ResNet50Crowd()
    x = torch.rand(size=(1,3,320,320))
    out = model(x)
    print('out.size: {}'.format(out.size()))
    compute_flops(model,img_size=(320,320),device='cpu')