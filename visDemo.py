"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/3/16-15:42
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import cv2
import argparse
import numpy as np
from PIL import Image
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import pandas as pd
from configs.config import get_parser

import torch
from torch import nn
import os.path as osp
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.utils import transforms as T
from models.load_model import create_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
normalizer = T.standard_transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
img_transformer = T.standard_transforms.Compose([
    T.standard_transforms.ToTensor(),
    normalizer
])

batch_size = 1
downsample_ratio = 1

img_root = r'/DCCUS_final/test_data/imgs'
gt_root = r'/DCCUS_final/test_data/npys'

save_dir = r'./vis_outputs/baseline/'

os.makedirs(save_dir, exist_ok=True)

def save_density_map(density_map,output_dir, fname='results.png'):
    np.seterr(divide='ignore', invalid='ignore')
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map,cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir,fname),density_map)

def cal_new_size_v2(im_h, im_w, min_size, max_size):
    rate = 1.0 * max_size / im_h
    rate_w = im_w * rate
    if rate_w > max_size:
        rate = 1.0 * max_size / im_w
    tmp_h = int(1.0 * im_h * rate / 16) * 16

    if tmp_h < min_size:
        rate = 1.0 * min_size / im_h
    tmp_w = int(1.0 * im_w * rate / 16) * 16

    if tmp_w < min_size:
        rate = 1.0 * min_size / im_w
    tmp_h = min(max(int(1.0 * im_h * rate / 16) * 16, min_size), max_size)
    tmp_w = min(max(int(1.0 * im_w * rate / 16) * 16, min_size), max_size)

    rate_h = 1.0 * tmp_h / im_h
    rate_w = 1.0 * tmp_w / im_w
    assert tmp_h >= min_size and tmp_h <= max_size
    assert tmp_w >= min_size and tmp_w <= max_size
    return tmp_h, tmp_w, rate_h, rate_w

def visDensityMap_and_Image_Overlap(imgName,image,density_map):
    img_log = image.detach().cpu().numpy()
    log_dir = "output/good_density/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    pred_density_write = 1. - density_map
    pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
    img = Image.fromarray(np.uint8(pred_density_write))
    img.save(f'{log_dir}/{imgName}_and_density_map.png')

    # log overlay
    log_dir = "output/good_pred/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    pred_density_write = pred_density_write / 255.
    img_write = 0.33 * np.transpose(img_log, (1, 2, 0)) + 0.67 * pred_density_write
    img = Image.fromarray(np.uint8(255 * img_write))
    img.save(f'{log_dir}/{imgName}_and_density_map.png')

def plot_ax(
        imgName,image,density_map,gt_map,
        gt_count,pred_count
):
    print("Visualize estimated density map")
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(imgName.replace('.jpg', ''))

    plt.gca().set_axis_off()  # 调整画布
    plt.margins(0, 0)  # 同时设置xy边距比例
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 设置x轴上的主刻度定位器。plt.NullLocator()表示删除刻度显示
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 设置y轴上的主刻度定位器。
    ax[0][0].imshow(np.asarray(image), cmap=CM.jet)
    ax[0][0].axis('off')
    ax[0][0].set_title(f'Input Image')

    plt.gca().set_axis_off()  # 调整画布
    plt.margins(0, 0)  # 同时设置xy边距比例
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 设置x轴上的主刻度定位器。plt.NullLocator()表示删除刻度显示
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 设置y轴上的主刻度定位器。
    ax[1][0].imshow(gt_map, cmap=CM.jet)
    ax[1][0].axis('off')
    ax[1][0].set_title(f'GT den {round(gt_count, 2)}')

    plt.gca().set_axis_off()  # 调整画布
    plt.margins(0, 0)  # 同时设置xy边距比例
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 设置x轴上的主刻度定位器。plt.NullLocator()表示删除刻度显示
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 设置y轴上的主刻度定位器
    ax[1][1].imshow(density_map, cmap=CM.jet)
    ax[1][1].axis('off')
    ax[1][1].set_title(f'pred {round(pred_count.item(), 2)}')

    ax[0][1].axis('off')
    plt.savefig(fname=os.path.join(save_dir,imgName.replace('jpg', 'png')), dpi=300)
    plt.close()


def divided_image_patch(inputs,h,w,crop_size, device):
    st_size = 1.0 * min(w, h)
    if st_size < crop_size:
        rr = 1.0 * crop_size / st_size
        wd = round(w * rr)
        ht = round(h * rr)
        st_size = 1.0 * min(wd, ht)
        inputs = inputs.resize((wd, ht), Image.BICUBIC)

    inputs = img_transformer(inputs)

    with torch.no_grad():
        # nputs = cal_new_tensor(inputs, min_size=args.crop_size)
        inputs = inputs.unsqueeze(0).to(device)

        b,c,h,w = inputs.size()

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
                        size=(h1 * downsample_ratio, w1 * downsample_ratio),
                        mode="bilinear",
                        align_corners=True,
                    ) / downsample_ratio * downsample_ratio
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
def visDensity_map01(model, crop_size):
    epoch_res = []
    for imgName in os.listdir(img_root):
        start_time = time()

        img_path = os.path.join(img_root, imgName)
        img = Image.open(img_path).convert("RGB")
        wd, ht = img.size
        print('width = {}  height = {}'.format(wd, ht))
        if wd >= crop_size or ht >= crop_size:
            pred_map = divided_image_patch(
                img,wd,ht,crop_size,device)
        else:
            image = img_transformer(img)
            image = image.unsqueeze(0).to(device)
            pred_map = model(image)
            print('here...')
        gt_count = np.sum(np.load(os.path.join(gt_root, imgName.replace('jpg', 'npy'))))
        # gt_count = np.sum(pd.read_csv(os.path.join(gt_root, os.path.splitext(imgName)[0] + '.csv'), sep=',', header=None).values)
        pred_count = pred_map.squeeze(0).squeeze(0).cpu().data.numpy().sum()
        #TODO 保存预测的密度图
        p_den_map = pred_map.squeeze(0).squeeze(0).detach().cpu().numpy()
        plt.figure()
        plt.axis('off')
        plt.text(
                 x=0.95,
                 y = 0.06,
                 s = f"{str(round(pred_count / 1000,1))}",
                 color='white',
                 transform=plt.gca().transAxes,
                 fontsize=32,
                 ha='right',  # 关键：右对齐
                 va='bottom',  # 关键：底部对齐
                 weight='bold')
        plt.imshow(p_den_map / 1000, cmap=CM.jet)
        plt.savefig(os.path.join(save_dir, imgName.replace('.jpg', '.png')))
        plt.close()
        # plot_ax('npy_' + imgName,img,
        #         density_map=pred_map.squeeze(0).squeeze(0).cpu().data.numpy(),
        #         gt_map=gt_dmap,gt_count=gt_count,
        #         pred_count = pred_count / 1000)


        print('inference time: {}s'.format(time() - start_time))
        print(f'detect {imgName} is finish!')

        res = gt_count - pred_count / 1000.0
        epoch_res.append(res)


    epoch_res = np.array(epoch_res)
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    return mae,mse



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    start_time = time()
    model, optim, start_epoch, best_mae, best_mse  = create_model(
        args,
        device
    )
    model.eval()
    print('best mae: {}  best mse: {}'.format(best_mae, best_mse))

    mae,mse = visDensity_map01(model, crop_size = args.crop_size)

    end_time = time()
    print('Test MAE: {:.4f}'.format(mae))
    print('Test MSE: {:.4f}'.format(mse))
    print('Test TIME: {}'.format(end_time - start_time))

    print('avg inference time/frame: {:.4f}'.format((end_time - start_time) / len(os.listdir(img_root))))
    pass
"""

    
"""