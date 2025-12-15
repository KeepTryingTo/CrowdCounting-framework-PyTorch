"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-10:17
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
from configs.config import get_parser
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

def main_worker(rank, device_ids, args):
    global device
    start_time = time.monotonic()
    # set current process's id
    args.rank = rank

    # set device
    if device_ids is not None and rank < len(device_ids):
        torch.cuda.set_device(device_ids[rank])
    else:
        torch.cuda.set_device(rank)

    if args.distributed:
        setup(args.rank, args.world_size)
        # torch.cuda.set_device(device_ids[rank])
        cudnn.benchmark = True
    # set current process's GPU corresponding id
    device = torch.device(f'cuda:{device_ids[rank]}')
    # TODO Create Model定义模型和优化器
    model, optim, start_epoch, best_mae, best_mse = create_model(args, device)

    if args.distributed:
        # DistributedDataParallel
        model = DDP(model, device_ids=[device_ids[rank]],
                    output_device=device_ids[rank],
                    find_unused_parameters=True)

    # TODO =====================Load Dataset ================================
    print("==> Load datasets")
    train_dataloader = get_train_loader(
        args, batch_size=args.batch_size,
        crop_size=args.crop_size_s
    )
    val_dataloader = get_test_loader(
        args, batch_size=1,
        crop_size=args.crop_size_t
    )

    if rank == 0:
        # TODO create log file
        current_date = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        args.logs_dir = os.path.join(args.logs_dir, args.data_type,
                                     's_' + str(args.crop_size_s) + '_t_' + str(args.crop_size_t) + '_' + current_date)
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    # TODO result log file, save MAE and MSE
    fp = open(file=os.path.join(str(args.logs_dir), 'mae_mse.txt'),
              mode='a+', encoding='utf-8')


    # TODO Evaluator
    if rank == 0:
        mse, mae = val_epoch(model, val_dataloader, 0, args.crop_size_t, args.d_ratio, device)
        print('mae: {}'.format(mae))
        print('mse: {}'.format(mse))
    # TODO ============================ Optimizer Define ======================
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # TODO statistic model's flops and parameters
    flops.compute_flops(model, img_size=(args.crop_size_s, args.crop_size_s), device=device)
    if optim is not None:
        optimizer.load_state_dict(optim)
    if args.lr_scheduler:
        if args.custom_lr_scheduler:
            lr_scheduler = CustomOneCycleLR(
                optimizer,
                max_lr=args.lr,
                total_steps=args.epochs * args.iters,  # total iterations steps（batch × epoch）
                pct_start=args.pct_start,  # 20% steps is learning ratio 👆
                div_factor=args.div_factor,  # initialize lr = max_lr / 10
                final_div_factor=args.final_div_factor  # final lr >= max_lr / 10
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
    if rank == 0:
        print('crop size s: {}'.format(args.crop_size_s))
        print('crop size t: {}'.format(args.crop_size_t))
        print('train dataset size: {}'.format(len(train_dataloader) * args.batch_size))
        print('val dataset size: {}'.format(len(val_dataloader)))

    best_mse_epochs = []
    best_mae_epochs = []

    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            print('==> start training epoch {} \t ==> learning rate = {}'.format(
                epoch, optimizer.param_groups[0]['lr']))
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
            device
        )
        # TODO save MAE and MSE of every epoch
        fp.write(str(round(best_mae, 3)) + ' ' + str(round(best_mse, 3)) + '\n')
        best_mae_epochs.append(best_mae)
        best_mse_epochs.append(best_mse)

        if rank == 0 and (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
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
            if rank == 0:
                print('\n * Finished epoch {:3d}  model mae: {:5.1f} mse: {:5.1f}  '
                      'best mae: {:5.1f} best mse: {:5.1f} {}\n'.
                      format(epoch, mae, mse, best_mae, best_mse, ' *' if is_best else ''))

    if args.distributed:
        # clean
        dist.destroy_process_group()
    fp.close()
    print('==> Test with the best model:')
    if rank == 0:
        with torch.no_grad():
            mse, mae = val_epoch(model, val_dataloader, 0, args.crop_size_t, args.d_ratio, device)
        end_time = time.monotonic()
        print('mae: {}  mse: {}'.format(mae, mse))
        print(f'Total running time: ', timedelta(seconds=end_time - start_time))

    torch.cuda.empty_cache()


def main(args):
    global device
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

    if args.distributed:
        print('distributed ......')

        # set current device
        if hasattr(args, 'device_ids') and args.device_ids:
            device_ids = setup_devices(args.device_ids, args.rank)
            args.world_size = len(device_ids)
            print('device_ids: {}'.format(device_ids))
        else:
            # default use all devices
            args.world_size = torch.cuda.device_count()
            device_ids = list(range(args.world_size))
            print(f"detect {args.world_size} GPU Numbers")

        assert 0 <= args.rank < args.world_size, f"无效的rank: {args.rank}"
        device = torch.device(f'cuda:{device_ids[args.rank]}')

        # add NCCL debug information
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

        args.world_size = args.world_size

        if args.world_size > 1:
            # distributed training
            mp.spawn(main_worker,
                     args=(device_ids, args),
                     nprocs=args.world_size,
                     join=True)

    else:
        # single GPU or CPU
        main_worker(0,(0,), args)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)

