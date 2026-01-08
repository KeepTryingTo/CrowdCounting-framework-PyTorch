"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-16:09
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
import os.path as osp
from collections import OrderedDict
from datetime import timedelta, datetime

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import flops
from utils.evaluator import val_epoch
from datasets.dataset.dataset import get_test_loader, get_train_loader

from losses.mse_loss import MSELoss
from losses.mae_loss import MAELoss

from configs.config_lr import scheduler_configs
from configs.config_ucf_cc_50 import get_parser
from utils.lr_schedular import CustomOneCycleLR,LRSchedulerManager
from trainer import train_epoch_one
from models.load_model import create_model
from utils.save_checkpoint import save_checkpoint_
from utils.distributed import setup,setup_devices
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

start_epoch = best_mAP = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def create_kfold_splits(args, num_folds=5):
    """
    创建5折交叉验证的数据分割
    """
    # 获取所有图像路径
    image_dir = os.path.join(args.data_dir, 'imgs')
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=args.seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        train_files = [image_files[i] for i in train_idx]
        val_files = [image_files[i] for i in val_idx]
        folds.append({
            'fold': fold_idx,
            'train_files': train_files,
            'val_files': val_files
        })

    return folds

def get_train_loader_with_fold(args, fold_info, batch_size, crop_size):
    # TODO 加载源域数据集和目标域数据集
    train_dataloader = get_train_loader(
        args, batch_size=args.batch_size,
        crop_size=args.crop_size_s,
        sub_folder_list=fold_info['train_files']
    )
    return train_dataloader


def get_val_loader_with_fold(args, fold_info, crop_size):
    val_dataloader = get_test_loader(
        args, batch_size=1,
        crop_size=args.crop_size_t,
        is_qnrf = args.is_qnrf,
        sub_folder_list=fold_info['val_files']
    )
    return val_dataloader

def main_worker(rank, device_ids, args, fold_info, fold_idx):
    global device
    start_time = time.monotonic()
    args.rank = rank

    # 先设置设备
    if device_ids is not None and rank < len(device_ids):
        torch.cuda.set_device(device_ids[rank])
    else:
        torch.cuda.set_device(rank)

    if args.distributed:
        setup(args.rank, args.world_size)
        torch.cuda.set_device(device_ids[rank])
        cudnn.benchmark = True
    device = torch.device(f'cuda:{device_ids[rank]}')

    # TODO Create Model
    model, optim, start_epoch, best_mae, best_mse = create_model(args, device)
    if args.distributed:
        # 使用DistributedDataParallel包装模型
        model = DDP(model, device_ids=[args.rank],
                    output_device=args.rank,
                    find_unused_parameters=True)

    # TODO =====================Load Dataset ================================
    print("==> Load datasets")
    # TODO load dataset
    train_dataloader = get_train_loader_with_fold(
        args, fold_info, args.batch_size, args.crop_size_s
    )
    val_dataloader = get_val_loader_with_fold(
        args, fold_info, args.crop_size_t
    )

    if args.rank == 0:
        # TODO create log file
        current_date = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        fold_logs_dir = os.path.join(
            args.logs_dir,
            f"fold_{fold_idx + 1}",
            args.data_type,
            f's_{args.crop_size_s}_t_{args.crop_size_t}_{current_date}'
        )
        if not os.path.exists(fold_logs_dir):
            os.makedirs(fold_logs_dir)

    fp = open(os.path.join(str(fold_logs_dir), 'mae_mse.txt'), 'a+', encoding='utf-8')

    # TODO Evaluator
    mse, mae = val_epoch(model, val_dataloader, 0, args.crop_size_t, args.d_ratio, device)
    print('mae: {}'.format(mae))
    print('mse: {}'.format(mse))
    # TODO ============================ Optimizer Define ======================
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # TODO 统计模型flops以及参数量
    flops.compute_flops(model, img_size=(args.crop_size_s, args.crop_size_s), device=device)
    if optim is not None:
        optimizer.load_state_dict(optim)
    if args.lr_scheduler:
        if args.custom_lr_scheduler:
            lr_scheduler = CustomOneCycleLR(
                optimizer,
                max_lr=args.lr,
                total_steps=args.epochs * args.iters,  # 总迭代步数（batch数×epoch数）
                pct_start=args.pct_start,  # 20%步数用于上升
                div_factor=args.div_factor,  # 初始lr = max_lr / 10
                final_div_factor=args.final_div_factor  # 最终lr >= max_lr / 10
            )
        else:
            lr_scheduler = LRSchedulerManager(
                optimizer,
                scheduler_config=scheduler_configs[args.scheduler_name]
            )
    else:
        lr_scheduler = None

    # TODO =======================Loss Define================================
    global_loss = MSELoss(lambda_global=args.weight_global, reduction='mean', device=device)
    count_loss = MAELoss(lambda_c=args.weight_cc, reduction='mean', device=device)

    # TODO ========================Trainer Define ============================
    print(f'Fold {fold_idx + 1} - Train samples: {len(train_dataloader.dataset)}')
    print(f'Fold {fold_idx + 1} - Val samples: {len(val_dataloader.dataset)}')
    print(f'Fold {fold_idx + 1} - Crop size s: {args.crop_size_s}')
    print(f'Fold {fold_idx + 1} - Crop size t: {args.crop_size_t}')

    best_mse_epochs = []
    best_mae_epochs = []

    for epoch in range(start_epoch, args.epochs):
        print(f'==> Fold {fold_idx + 1} - Epoch {epoch}: learning rate = {optimizer.param_groups[0]["lr"]}')
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        model.train()
        train_epoch_one(
            args,
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            epoch,
            global_loss,
            count_loss,
            args.d_ratio,
            device
        )
        # TODO 保存每一个epochs的mae和mse
        fp.write(str(round(best_mae, 3)) + ' ' + str(round(best_mse, 3)) + '\n')
        best_mae_epochs.append(best_mae)
        best_mse_epochs.append(best_mse)

        if args.rank == 0 and (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            print('==> start evaluate\n')
            with torch.no_grad():
                mse, mae = val_epoch(
                    model,
                    val_dataloader,
                    epoch,
                    args.crop_size_t,
                    args.d_ratio,
                    device
                )  # TODO 14.8
            is_best = (mae < best_mae or mse < best_mse)

            best_mae, best_mse = save_checkpoint_(
                args,
                model,
                epoch,
                best_mae,
                best_mse,
                optimizer,
                is_best,
                mae,
                mse
            )

            torch.cuda.empty_cache()
            print('\n * Finished epoch {:3d}  model mae: {:5.1f} mse: {:5.1f}  '
                  'best mae: {:5.1f} best mse: {:5.1f} {}\n'.
                  format(epoch, mae, mse, best_mae, best_mse, ' *' if is_best else ''))

    if args.distributed:
        # 清理分布式环境
        dist.destroy_process_group()

    fp.close()
    print('==> Test with the best model:')
    with torch.no_grad():
        mse, mae = val_epoch(model, val_dataloader, 0, args.crop_size_t, args.d_ratio, device)
    end_time = time.monotonic()
    print('mae: {}  mse: {}'.format(mae, mse))
    print(f'Total running time: ', timedelta(seconds=end_time - start_time))

    return best_mae, best_mse

def main(args):
    if args.data_type.lower() == 'qnrf':
        args.crop_size_s = 512
        args.crop_size_t = 3584
        args.is_qnrf = True
    elif args.data_type.lower() == 'nwpu':
        args.crop_size_s = 384
        args.crop_size_t = 2048
    elif args.data_type.lower() == 'sha':
        args.crop_size_s = 512
        args.crop_size_t = 2048
    elif args.data_type.lower() == 'shb':
        args.crop_size_s = 512
        args.crop_size_t = 3584
    elif args.data_type.lower() == 'custom':
        args.crop_size_s = 512
        args.crop_size_t = 2048
    elif args.data_type.lower() == 'jhu':
        args.crop_size_s = 512
        args.crop_size_t = 2048
    elif args.data_type.lower == 'ucf_cc_50':
        args.crop_size_s = 320
        args.crop_size_t = 1024
    else:
        raise NotImplementedError

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # print parameters
    table = PrettyTable()
    table.field_names = ['Argument', 'Value']
    for arg, value in vars(args).items():
        table.add_row([arg, value])
    print(table)

    # TODO 创建5折交叉验证分割
    folds = create_kfold_splits(args, num_folds=5)

    start_time = time.time()

    # TODO 存储每个fold的结果
    all_fold_results = []

    for fold_idx, fold_info in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"Starting Fold {fold_idx + 1}/5")
        print(f"{'=' * 60}")

        fold_start_time = time.time()

        if args.distributed:

            print('distributed ......')

            # 设置指定设备
            if hasattr(args, 'device_ids') and args.device_ids:
                device_ids = setup_devices(args.device_ids, args.rank)
                args.world_size = len(device_ids)
                print('device_ids: {}'.format(device_ids))
            else:
                # 默认使用所有可用设备
                args.world_size = torch.cuda.device_count()
                device_ids = list(range(args.world_size))
                print(f"detect {args.world_size} GPU Numbers")

            assert 0 <= args.rank < args.world_size, f"无效的rank: {args.rank}"
            device = torch.device(f'cuda:{device_ids[args.rank]}')

            # 添加NCCL调试信息
            os.environ['NCCL_DEBUG'] = 'INFO'
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

            args.world_size = args.world_size

            if args.world_size > 1:
                # 多GPU分布式训练
                mp.spawn(main_worker,
                         args=(device_ids, args, fold_info, fold_idx),
                         nprocs=args.world_size,
                         join=True)
        else:
            # 单GPU训练（保持原有逻辑）
            best_mae, best_mse = main_worker(0, (0, ), args, fold_info, fold_idx)
            fold_time = time.time() - fold_start_time

            all_fold_results.append({
                'fold': fold_idx + 1,
                'best_mae': best_mae,
                'best_mse': best_mse,
                'time': fold_time
            })

            print(f'Fold {fold_idx + 1} completed in {timedelta(seconds=fold_time)}')
            print(f'Fold {fold_idx + 1} - Best MAE: {best_mae:.1f}, Best MSE: {best_mse:.1f}')

    # 计算并输出交叉验证结果
    print(f"\n{'=' * 70}")
    print("5-Fold Cross Validation Final Results")
    print(f"{'=' * 70}")

    mae_values = [result['best_mae'] for result in all_fold_results]
    mse_values = [result['best_mse'] for result in all_fold_results]
    time_values = [result['time'] for result in all_fold_results]

    mean_mae = np.mean(mae_values)
    std_mae = np.std(mae_values)
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    total_time = np.sum(time_values)

    # 输出详细结果
    result_table = PrettyTable()
    result_table.field_names = ['Fold', 'MAE', 'MSE', 'Time']
    for result in all_fold_results:
        result_table.add_row([
            result['fold'],
            f"{result['best_mae']:.1f}",
            f"{result['best_mse']:.1f}",
            f"{timedelta(seconds=result['time'])}"
        ])

    result_table.add_row(['Mean', f"{mean_mae:.1f} ± {std_mae:.1f}", f"{mean_mse:.1f} ± {std_mse:.1f}",
                          f"{timedelta(seconds=total_time)}"])

    print(result_table)

    # 保存总体结果
    overall_result_file = os.path.join(args.logs_dir, 'cross_validation_results.txt')
    with open(overall_result_file, 'w', encoding='utf-8') as f:
        f.write("5-Fold Cross Validation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: {args.data_type}\n")
        f.write(f"Total Time: {timedelta(seconds=total_time)}\n\n")
        f.write(str(result_table) + "\n\n")
        f.write("Detailed Results:\n")
        for result in all_fold_results:
            f.write(
                f"Fold {result['fold']}: MAE={result['best_mae']:.1f}, MSE={result['best_mse']:.1f}, Time={timedelta(seconds=result['time'])}\n")

    end_time = time.monotonic()
    print(f'\nTotal running time: {timedelta(seconds=end_time - start_time)}')
    print(f'Results saved to: {overall_result_file}')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)

