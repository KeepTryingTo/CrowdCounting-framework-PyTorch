from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from datasets.utils import transforms as T

from scipy import io as sio
import torch
import random
import torchvision.transforms.functional as F

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map

    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    # TODO 得到标注点的Y坐标和X坐标
    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
    # TODO 将坐标映射到一维的坐标向量
    """
    points_np = [1,2] => [3,4] => x = 3,y = 4
    h = 5,w = 5
    4 * w + x = 19
    """
    p_index = torch.from_numpy(p_h * im_width + p_w).to(torch.int64)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(dim=0,
                                                                  index=p_index,
                                                                  src=torch.ones(im_width * im_height)
                                                                  ).view(im_height, im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map

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

class Preprocessor_tran(Dataset):
    def __init__(self,args, root=None, main_transform=None,
                 img_transform=None, gt_transform=None,
                 crop_size = 320,d_ratio = 1,
                 sub_folder_list = None,
                 is_target = False):
        super(Preprocessor_tran, self).__init__()
        # if is_target is False:
        #     root = os.path.join(root, "train_data")

        self.args = args
        self.root = root
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.crop_size = crop_size
        self.d_ratio = d_ratio



        if sub_folder_list:
            self.img_list = sub_folder_list
        else:
            self.img_list = os.listdir(os.path.join(root, 'imgs'))

        self.base_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4,
                    saturation=0.4, hue=0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])
        self.random_flip = T.Compose([
            T.RandomHorizontallyFlip()
        ])

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
        
        #TODO  den = pd.read_csv(os.path.join(self.root+'/den', os.path.splitext(fname)[0] + '.csv'), sep=',', header=None).values
        den = np.load(os.path.join(self.root + '/npys',os.path.splitext(fname)[0] + '.npy'))
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        # TODO Padding 根据高宽对图像进行填充
        w, h = img.size
        st_size = 1.0 * min(w, h)
        # TODO 如果小于指定裁剪大小就填充
        if st_size < min(self.crop_size, self.crop_size):
            # (left, top, right, bottom), h = crop_size_h, w = crop_size_w
            padding, h, w = get_padding(h, w, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            crop_tran = T.RandomCrop(self.crop_size, padding=padding)
            img, den = crop_tran(img, den)
        else:
            crop_tran = T.RandomCrop(self.crop_size, padding=0)
            img, den = crop_tran(img, den)

        #TODO flip
        if self.main_transform is not None:
            img, den = self.main_transform(img,den)

        #TODO 将图像转换为tensor同时进行归一化操作
        if self.img_transform is not None:
            img = self.img_transform(img)
        #TODO 对密度图进行处理，比如乘以1000
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        return img, den

    def train_transform(self, img, keypoints,den):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # TODO 如果图像的高宽最小值小于裁剪的大小 resize the image to fit the crop size
        if st_size < self.crop_size:
            rr = 1.0 * self.crop_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.crop_size, print(wd, ht)
        assert len(keypoints) >= 0
        # TODO 选择裁剪的起始位置和终止位置
        i, j, h, w = random_crop(ht, wd, self.crop_size, self.crop_size)
        # TODO 裁剪图像
        img = F.crop(img, i, j, h, w)
        den = F.crop(den, i, j, h, w)
        # TODO 如果标注点大于0 的话就对图像中的标注坐标进行转换
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            # TODO 过滤掉不满足要求的点标注
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])
        # TODO 将坐标离散化：二维 => 到一维向量 => 二维向量
        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        # TODO 根据下采样比率
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio,
                                           down_w, self.d_ratio]
                                          ).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)
        #TODO 同时对密度图也需要进行裁剪

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                den = F.hflip(den)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                den = F.hflip(den)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return (img,den, torch.from_numpy(keypoints.copy()).float(),
                torch.from_numpy(gt_discrete.copy()).float())
