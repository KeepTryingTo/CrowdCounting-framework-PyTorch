"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/4/3-14:45
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


def vis_Gt(npy_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for npyName in os.listdir(npy_root):
        gt_path = os.path.join(npy_root, npyName)
        gt_dmap = np.load(gt_path)
        gt_count = np.sum(gt_dmap)
        plt.figure()
        plt.axis('off')
        plt.text(
            x=0.95,
            y=0.06,
            s=f"{str(round(gt_count, 1))}",
            color='white',
            transform=plt.gca().transAxes,
            fontsize=32,
            ha='right',  # 关键：右对齐
            va='bottom',  # 关键：底部对齐
            weight='bold')
        plt.imshow(gt_dmap, cmap=CM.jet)
        plt.savefig(os.path.join(save_dir, npyName.replace('npy','png')))
        plt.close()
        print(f'gt {npyName} is finished!')


if __name__ == '__main__':
    vis_Gt(None, None)
    pass
