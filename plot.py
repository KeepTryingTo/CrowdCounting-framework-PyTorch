"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-14:12
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 指定字体文件路径（替换为你的字体路径）
# font_path = 'C:/Windows/Fonts/msyh.ttc'  # Windows示例
# font_manager.fontManager.addfont(font_path)
# plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

def read_file(root):
    # 读取文件数据
    def read_data(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        mae, mse = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                mae.append(float(parts[0]))
                mse.append(float(parts[1]))
        return np.array(mae), np.array(mse)

    return read_data(root)

def plot_mae(
        paths,
        labels
):
    if paths == None or labels == None:
        print('paths and labels are can not None')
        return
    #TODO 读取两个文件mae, mse
    maes = []
    for path in paths:
        mae, mse = read_file(path)
        #TODO 去掉最开始的100000
        mae = mae[1:]
        maes.append(mae)

    x = [i for i in range(len(maes[0]))]

    #TODO 绘制图形
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

    for i, (mae, label) in enumerate(zip(maes, labels)):
        color = colors[i % len(colors)]
        plt.plot(x, mae, color=color, label=f'{label}', linewidth=2)

    #TODO 添加标题和标签
    plt.title(f'MAE METRICS', fontsize=14)
    plt.xlabel('EPOCHS', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示图形
    plt.savefig("./outputs/result.png")
    plt.show()

if __name__ == '__main__':
    plot_mae(
        paths=[r'/home/ff/myProject/KGT/myProjects/myProjects/CrowdCounting-framework-PyTorch/weights/logs/sha/s_512_t_2048_2025-11-21_21-21-46/mae_mse.txt'],
        labels=['MAE']
    )
