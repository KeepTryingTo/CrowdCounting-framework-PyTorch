"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-10:18
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""


import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TQDM_COLUMNS"] = "80"  # 指定列数

import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
from models.load_model import create_model
from datasets.dataset.dataset import get_test_loader
from configs.config import get_parser

device = 'cpu' if torch.cuda.is_available() else 'cpu'

def divided_image_patch(inputs,model,d_ratio,b, h,w,crop_size, device):
    crop_imgs, crop_masks = [], []
    # TODO 首先对图像进行裁剪，将裁剪之后的图像patch输入到模型中检测
    rh, rw = crop_size, crop_size
    mask = torch.zeros([b, 1, h, w]).to(device)
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
            crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
            mask[:, :, gis:gie, gjs:gje] += 1

    crop_imgs = torch.cat(crop_imgs, dim=0)

    crop_preds = []
    nz, bz = crop_imgs.size(0), 1
    for i in range(0, nz, bz):
        gs, gt = i, min(nz, i + bz)
        crop_pred = model(crop_imgs[gs:gt])

        _, _, h1, w1 = crop_pred.size()
        crop_pred = (
                F.interpolate(
                    crop_pred,
                    size=(h1 * d_ratio, w1 * d_ratio),
                    mode="bilinear",
                    align_corners=True,
                ) / d_ratio * d_ratio
        )

        crop_preds.append(crop_pred)
    crop_preds = torch.cat(crop_preds, dim=0)

    # TODO splice them to the original size
    idx = 0
    pred_map = torch.zeros([b, 1, h, w]).to(device)
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
            pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
            idx += 1
    outputs = pred_map / mask
    return outputs

def val_epoch(model,
              dataloader,
              epoch,
              crop_size,
              d_ratio,
              device):
    epoch_res = []
    model.eval()
    print('valing: ')
    for step,(inputs, gt) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = inputs.to(device)
            b, c, h, w = inputs.size()

            if h >= crop_size or w >= crop_size:
                outputs = divided_image_patch(inputs,model,d_ratio,
                                              b,h,w,crop_size,device)
            else:
                outputs = model(inputs)
            res = (abs(torch.sum(gt).item() - torch.sum(outputs).item())) / 1000.0

            epoch_res.append(res)
        # print(f'{step} gt count: {torch.sum(count).item() / 1000.0}  pred count :{torch.sum(outputs).item() / 1000.0}')
    epoch_res = np.array(epoch_res)
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))

    print('val [{}]  mae: {:.3f}  mse: {:.3f}'.format(epoch, mae, mse))

    return mse, mae


if __name__ == '__main__':

    args = get_parser()

    model, optim, start_epoch, best_mae, best_mse = create_model(args, device)
    model.to(device)
    model.eval()

    test_loader = get_test_loader(args, batch_size=1)

    mse, mae = val_epoch(
        model=model,
        dataloader=test_loader,
        epoch=0,
        crop_size=args.crop_size,
        d_ratio=1,
        device=device
    )

