"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-13:53
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn

class MAELoss(nn.Module):
    def __init__(self, lambda_c = 1.0, reduction = 'mean', device = 'cpu'):
        super().__init__()
        self.lambda_c = lambda_c
        self.reduction = reduction
        self.device = device
        self.mae_loss = nn.L1Loss(self.reduction).to(self.device)

    def forward(self, pred, gt):
        return self.lambda_c * self.mae_loss(pred, gt)