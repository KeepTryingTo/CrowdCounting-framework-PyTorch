"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-14:03
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import os.path as osp

def save_checkpoint_(
        args,
        model,
        epoch,
        best_mae,
        best_mse,
        optimizer,
        is_best,
        mae,
        mse
):
    # 处理DDP包装的模型
    if hasattr(model, 'module'):
        # 如果是DDP模型，提取原始模型
        model_state_dict = model.module.state_dict()
    else:
        # 如果是普通模型，直接使用
        model_state_dict = model.state_dict()
    saved_model = {
        'state_dict': model_state_dict,
        'epoch': epoch,
        'mae': best_mae,
        'mse': best_mse,
        'optim': optimizer.state_dict()
    }

    if is_best:
        best_mae = mae
        best_mse = mse
        saved_model['mae'] = best_mae
        saved_model['mse'] = best_mse
        torch.save(saved_model,
                   osp.join(args.logs_dir,
                            f'bestmodel_{str(round(best_mae, 3))}.pth.tar'))
        torch.save(saved_model, osp.join(args.logs_dir, 'latestmodel.pth.tar'))

    return best_mae, best_mse