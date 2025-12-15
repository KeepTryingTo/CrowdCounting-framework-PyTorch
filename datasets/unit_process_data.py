"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-15:12
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import cv2
import glob
import shutil
from pathlib import Path
from typing import List, Set, Dict


from torch import nn
from datasets.copy import FileSynchronizer
from datasets.dmap_process.dmap_for_NWPU import main_nwpu
from datasets.dmap_process.dmap_for_QNRF import main_qnrf
from datasets.dmap_process.dmap_for_SHHA import main_shaa
from datasets.dmap_process.dmap_for_SHHB import main_shab
from datasets.dmap_process.dmap_for_jhu import main_jhu
from datasets.dmap_process.dmap_for_ucf_cc_50 import main_ucf_cc_50


class ManagerDataset(nn.Module):
    def __init__(self, dataset_name, root_dir, save_dir):
        super().__init__()
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.save_dir = os.path.join(save_dir, self.dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)


    def forward(self):
        # 生成密度图，并将密度图保存到指定的save_dir文件目录下
        if self.dataset_name == 'nwpu':
            main_nwpu(self.root_dir, self.save_dir)
        elif self.dataset_name == 'qnrf':
            main_qnrf(self.root_dir, self.save_dir)
        elif self.dataset_name == 'sha':
            main_shaa(self.root_dir, self.save_dir)
        elif self.dataset_name == 'shb':
            main_shab(self.root_dir, self.save_dir)
        elif self.dataset_name == 'ucf_cc_50':
            # 由于ucf_cc_50是没有明确的测试集和训练集之分的，因此训练的时候采用的是K折交叉验证
            # 因此，这里不需要创建训练集和测试集对应的文件目录
            self.ucf_cc_50_imgs = os.path.join(str(self.save_dir), 'imgs')
            self.ucf_cc_50_npys = os.path.join(str(self.save_dir), 'npys')
            os.makedirs(self.ucf_cc_50_imgs, exist_ok=True)
            os.makedirs(self.ucf_cc_50_npys, exist_ok=True)
            main_ucf_cc_50(self.root_dir, self.save_dir)
        elif self.dataset_name == 'jhu':
            main_jhu(self.root_dir, self.save_dir)

        # 将原始图像也复制到指定的os.path.join(self.save_dir, 'imgs')路径下，
        # 统一处理所有类型的数据集，然后后面就可以一键训练了
        if self.dataset_name != 'ucf_cc_50':
            self.img_train_root_save_dir = os.path.join(str(self.save_dir), 'train', 'imgs')
            self.img_test_root_save_dir = os.path.join(str(self.save_dir), 'test', 'imgs')

            os.makedirs(self.img_train_root_save_dir, exist_ok=True)
            os.makedirs(self.img_test_root_save_dir, exist_ok=True)

        if self.dataset_name == 'jhu':
            for phrase in ['train', 'val']:
                img_root = os.path.join(self.root_dir, phrase, 'images')
                copyor = FileSynchronizer(
                    target_dir=self.img_train_root_save_dir if phrase == 'train' else self.img_test_root_save_dir,
                    source_dir=img_root
                )
                copyor.sync_files(
                    file_extensions=['.jpg'],
                    dry_run=False
                )
        elif self.dataset_name == 'nwpu':
            img_root = os.path.join(self.root_dir, 'images')
            for phase in ['train', 'val']:
                with open(os.path.join(self.root_dir, 'mats', '{}.txt'.format(phase))) as f:
                    lines = f.readlines()
                    for i in lines:
                        i = i.strip().split(' ')[0]
                        im_path = os.path.join(img_root, i + '.jpg')
                        if phase == 'train':
                            target_path = os.path.join(self.img_train_root_save_dir, i + '.jpg')
                            shutil.copy2(im_path, target_path)
                        elif phase == 'test':
                            target_path = os.path.join(self.img_test_root_save_dir, i + '.jpg')
                            shutil.copy2(im_path, target_path)

        elif self.dataset_name == 'sha' or self.dataset_name == 'shb':
            for mod in ['train', 'test']:
                img_root = os.path.join(self.root_dir, mod + '_data', 'images')
                copyor = FileSynchronizer(
                    target_dir=self.img_train_root_save_dir if mod == 'train' else self.img_test_root_save_dir,
                    source_dir=img_root
                )
                copyor.sync_files(
                    file_extensions=['.jpg'],
                    dry_run=False
                )

        elif self.dataset_name == 'qnrf':
            for phrase in ['Train', 'Test']:
                img_root = os.path.join(self.root_dir, phrase)
                copyor = FileSynchronizer(
                    target_dir=self.img_train_root_save_dir if phrase == 'Train' else self.img_test_root_save_dir,
                    source_dir=img_root
                )
                copyor.sync_files(
                    file_extensions=['.jpg', '.csv'],
                    dry_run=False
                )

        elif self.dataset_name == 'ucf_cc_50':
            img_root = os.path.join(self.root_dir, 'images')
            copyor = FileSynchronizer(
                target_dir=self.ucf_cc_50_imgs,
                source_dir=img_root
            )
            copyor.sync_files(
                file_extensions=['.jpg'],
                dry_run=False
            )


if __name__ == '__main__':
    managerDataset = ManagerDataset(
        dataset_name='ucf_cc_50',
        root_dir=r'/home/ff/myProject/KGT/myProjects/myDataset/UCF_CC_50',
        save_dir=r'/home/ff/myProject/KGT/myProjects/myProjects/CrowdCounting-framework-PyTorch/datasets/process_data'
    )

    managerDataset()
