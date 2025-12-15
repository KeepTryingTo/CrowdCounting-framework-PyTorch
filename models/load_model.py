"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-10:29
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from collections import OrderedDict
from models.crowd_vgg import CrowdVGG
from models.crowd_resnet import CrowdResNet

def create_model(args, device):
    if args.backbone_m == "crowdvgg":
        model = CrowdVGG(pretrained=True)
        print('load crowdvgg model is done ...')
    elif args.backbone_m == "crowdresnet":
        model = CrowdResNet(pretrained=True)
        print('load crowdresnet model is done ...')
    model.to(device)
    optim = None
    best_mae = 100000
    best_mse = 100000
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            if 'total_ops' in key or 'total_params' in key:
                continue
            else:
                new_state_dict[key] = checkpoint['state_dict'][key]
        model.load_state_dict(new_state_dict)
        best_mae = checkpoint['mae']
        best_mse = checkpoint['mse']
        start_epoch = checkpoint['epoch']
        optim = checkpoint['optim']
    return model, optim, start_epoch, best_mae, best_mse
