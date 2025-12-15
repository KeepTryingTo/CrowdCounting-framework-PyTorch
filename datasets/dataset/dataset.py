"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/7/25-20:39
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""


import os
import torch
from torch.utils.data import DataLoader

from datasets.utils import IterLoader
from datasets.utils.preprocessor import Preprocessor
from datasets.utils.preprocessor_tran import Preprocessor_tran
from datasets.utils import transforms as T


def get_train_loader(
         args,
         batch_size,
         workers=0,
         crop_size = 256,
         d_ratio = 1,
         sub_folder_list = None,
         is_target = False
):

    normalizer = T.standard_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transformer = T.Compose([
         T.RandomHorizontallyFlip()
    ])

    img_transformer = T.standard_transforms.Compose([
        T.standard_transforms.ToTensor(),
        normalizer
    ])
    gt_transformeer = T.standard_transforms.Compose([
        T.LabelNormalize(1000.)
    ])

    train_dataset = Preprocessor_tran(
        args = args,
        root = args.data_dir,
        main_transform = train_transformer,
        img_transform = img_transformer,
        gt_transform = gt_transformeer,
        crop_size = crop_size,
        d_ratio = d_ratio,
        sub_folder_list=sub_folder_list,
        is_target=False
    )
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=False,
        drop_last=True
    )

    return train_loader

def get_test_loader(
        args,
        batch_size,
        workers = 0,
        crop_size = 256,
        is_qnrf = False,
        sub_folder_list = None,
        is_target = True
):
    normalizer = T.standard_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    test_transformer = None
    img_transformer = T.standard_transforms.Compose([
        T.standard_transforms.ToTensor(),
        normalizer
    ])
    gt_transformer = T.standard_transforms.Compose([
        T.LabelNormalize(1000.)
    ])

    test_dataset = Preprocessor(
        root=args.test_data_dir,
        main_transform=test_transformer,
        img_transform=img_transformer,
        gt_transform=gt_transformer,
        unit_size=crop_size,
        is_qnrf = is_qnrf,
        sub_folder_list=sub_folder_list,
        is_target=True
    )
    test_sampler = None
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False
    )

    return test_loader