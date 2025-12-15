"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-20:26
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import os
import torch
import torch.distributed as dist

import time
import socket
import random


def get_available_port(start_port=29500, max_attempts=100):
    """动态获取可用端口"""
    for attempt in range(max_attempts):
        port = start_port + attempt
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError("无法找到可用端口")


def setup(rank, world_size):
    """改进的动态端口初始化"""
    if rank == 0:
        available_port = get_available_port()
        # 将端口号保存到文件，供其他进程读取
        with open('distributed_port.txt', 'w') as f:
            f.write(str(available_port))
        os.environ['MASTER_PORT'] = str(available_port)
        print(f"使用端口: {available_port}")
    else:
        # 等待端口文件创建
        while not os.path.exists('distributed_port.txt'):
            time.sleep(0.1)
        # 读取端口号
        with open('distributed_port.txt', 'r') as f:
            port = f.read().strip()
        os.environ['MASTER_PORT'] = port

    os.environ['MASTER_ADDR'] = 'localhost'

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    # torch.cuda.set_device(rank)

def setup_devices(device_ids, rank):
    """
    修改设备设置函数，考虑rank参数
    """
    if isinstance(device_ids, str):
        device_ids = [int(x.strip()) for x in device_ids.split(',')]
    # 这里设置了CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    # 确保rank在设备范围内
    if rank >= len(device_ids):
        raise ValueError(f"rank {rank} 超出设备列表范围 {device_ids}")

    # 如果指定了rank，设置当前进程的设备
    # if rank is not None and rank < len(device_ids):
    #     torch.cuda.set_device(device_ids[rank])
    return device_ids