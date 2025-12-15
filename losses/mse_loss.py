"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-13:50
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self, lambda_global = 1.0, reduction = 'mean', device = 'cpu'):
        super().__init__()
        self.lambda_global = lambda_global
        self.reduction = reduction
        self.device = device
        self.mse_loss = nn.MSELoss(self.reduction).to(self.device)

    def forward(self, pred, gt):
        return self.lambda_global * self.mse_loss(pred, gt)