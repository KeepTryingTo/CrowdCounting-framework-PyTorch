from __future__ import absolute_import
import os
import random
import math
import numpy as np
import pandas as pd
import os.path as osp
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset


def get_padding(h, w, new_h, new_w):
    if h >= new_h:
        top = 0
        bottom = 0
    else:
        dh = new_h - h
        top = dh // 2
        bottom = dh // 2 + dh % 2
        h = new_h
    if w >= new_w:
        left = 0
        right = 0
    else:
        dw = new_w - w
        left = dw // 2
        right = dw // 2 + dw % 2
        w = new_w

    return (left, top, right, bottom), h, w

class Preprocessor(Dataset):
    def __init__(self, unit_size = 16, root=None, main_transform=None,
                 img_transform=None, gt_transform=None, is_qnrf = False,
                 sub_folder_list = None,is_target = False):
        super(Preprocessor, self).__init__()
        # if is_target:
        #     root = os.path.join(root, "test_data")
        if sub_folder_list:
            self.img_list = sub_folder_list
        else:
            self.img_list = os.listdir(os.path.join(root, 'imgs'))
        self.root = root
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.unit_size = unit_size
        self.is_qnrf = is_qnrf

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.img_list[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root+'/imgs', fname)
        
        img = Image.open(fpath).convert('RGB')

        # TODO 将图像缩放到指定的比例
        if self.unit_size > 0:
            w, h = img.size
            new_w = (w // self.unit_size + 1) * self.unit_size if w % self.unit_size != 0 else w
            new_h = (h // self.unit_size + 1) * self.unit_size if h % self.unit_size != 0 else h
            # 对图像进行填充至指定图像大小
            padding, h, w = get_padding(h, w, new_h, new_w)

            img = F.pad(img, padding)

        if self.is_qnrf:
            den = pd.read_csv(os.path.join(self.root+'/npys', os.path.splitext(fname)[0] + '.csv'), sep=',', header=None).values
        else:
            den = np.load(os.path.join(self.root + '/npys', os.path.splitext(fname)[0] + '.npy'))
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        if self.main_transform is not None:
            img, den = self.main_transform(img,den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        return img, den