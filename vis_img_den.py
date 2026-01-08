"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/12/10-18:28
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

"""
    下面实现的方法主要是将原图和模型预测的密度进行一个叠加，
    方便看清楚原图的人群位置和预测的结果，对比看一下效果怎么
    样
"""

import numpy as np
import cv2
import os
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm as CM

from collections import OrderedDict

import torch
import torch.nn.functional as F
from models.load_model import create_model
from configs.config import get_parser
class DensityMapVisualizer:
    def __init__(self, colormap='jet', use_opencv=True):
        """
        colormap: 颜色映射名称
        use_opencv: 是否使用OpenCV（更快）
        """
        self.colormap = colormap
        self.use_opencv = use_opencv

        if use_opencv:
            # OpenCV颜色映射映射
            self.cv_colormap = {
                'jet': cv2.COLORMAP_JET,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'hot': cv2.COLORMAP_HOT,
                'cool': cv2.COLORMAP_COOL,
                'spring': cv2.COLORMAP_SPRING,
                'summer': cv2.COLORMAP_SUMMER,
                'autumn': cv2.COLORMAP_AUTUMN,
                'winter': cv2.COLORMAP_WINTER
            }

    def normalize_density(self, density, method='minmax', eps=1e-8):
        """
        归一化密度图
        method: 'minmax' 或 'log'
        """
        density = density.squeeze()  # 去除单通道维度

        if method == 'minmax':
            # 最小-最大归一化（比较常用的方法）
            d_min, d_max = density.min(), density.max()
            if d_max - d_min < eps:
                return np.zeros_like(density)
            return (density - d_min) / (d_max - d_min)

        elif method == 'log':
            # 对数归一化
            density_log = np.log(density + 1)
            d_max = density_log.max()
            if d_max < eps:
                return np.zeros_like(density)
            return density_log / d_max

        elif method == 'percentile':
            # 百分位归一化，排除异常值
            p_low, p_high = np.percentile(density, [2, 98])
            density_clipped = np.clip(density, p_low, p_high)
            return (density_clipped - p_low) / (p_high - p_low + eps)

        else:
            raise ValueError(f"不支持的归一化方法: {method}")
    def save_density_map(self,img, pred_density):

        pred_density = pred_density / pred_density.max()
        pred_density_write = 1. - pred_density
        pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
        pred_density_write = pred_density_write / 255

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        heatmap_pred = 0.33 * img + 0.67 * pred_density_write
        heatmap_pred = heatmap_pred / heatmap_pred.max()
        query_img = cv2.cvtColor((heatmap_pred * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

        return query_img

    def visualize_comparison(self,
                             image,
                             density_map,
                             gt_density_map,
                             alpha=0.5,
                             normalize_method='log',
                             save_path=None):
        """
        :param image: opencv读取的图像格式
        :param density_map:  模型预测的密度图
        :param alpha: 融合的权重因子
        :param normalize_method: 预测密度图的归一化方式，比如log，min-max，percentile
        :param save_path: 保存图像的路径
        :return:
        """

        overlay = self.save_density_map(
            img=image, pred_density=density_map
        )

        fig, axes = plt.subplots(2, 3, figsize=(24, 16))

        plt.subplots_adjust(
            left=0.05,  # 左边距
            right=0.095,  # 右边距
            bottom=0.05,  # 底边距
            top=0.092,  # 顶边距
            wspace=0.01,  # 水平间距（从0.3减小到0.1）
            hspace=0.015  # 垂直间距（从0.5减小到0.15）
        )

        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # 原始密度图
        im1 = axes[0, 1].imshow(density_map.squeeze(), cmap=CM.jet)
        axes[0, 1].set_title('Density Map (Raw)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        axes[0, 1].text(
            x=0.95,
            y=0.06,
            s=f"{str(round(np.sum(density_map) / 1000, 1))}",
            color='white',
            transform=axes[0, 1].transAxes,
            fontsize=32,
            ha='right',  # 关键：右对齐
            va='bottom',  # 关键：底部对齐
            weight='bold')

        # 真实密度图
        im2 = axes[0, 2].imshow(gt_density_map, cmap=CM.jet)
        axes[0, 2].set_title(f'GT Density Map')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        axes[0, 2].text(
            x=0.95,
            y=0.06,
            s=f"{str(round(np.sum(gt_density_map), 1))}",
            color='white',
            transform=axes[0, 2].transAxes,
            fontsize=32,
            ha='right',  # 关键：右对齐
            va='bottom',  # 关键：底部对齐
            weight='bold')

        # 叠加图
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title(f'Overlay (alpha={alpha})')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        axes[1, 0].text(
            x=0.95,
            y=0.06,
            s=f"{str(round(np.sum(density_map) / 1000, 1))}",
            color='white',
            transform=axes[1, 0].transAxes,
            fontsize=32,
            ha='right',  # 关键：右对齐
            va='bottom',  # 关键：底部对齐
            weight='bold')
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")

        return overlay


def process(img_path, density_pred, gt_density_map, save_path):
    """
    :param img_path: 图像路径
    :param density_pred: 输入[1, 1, H, W] [Tensor格式]
    :return:
    """
    # 创建可视化器
    visualizer = DensityMapVisualizer(colormap='jet', use_opencv=True)

    image = cv2.imread(img_path)

    if len(density_pred.shape) == 4:  # [1, 1, H, W]
        density_pred = density_pred.squeeze(0).squeeze(0)
    elif len(density_pred.shape) == 3:  # [1, H, W] 或 [H, W, 1]
        density_pred = density_pred.squeeze()

    if isinstance(density_pred, torch.Tensor):
        density_pred = density_pred.cpu().data.numpy()
    # 显示对比图
    visualizer.visualize_comparison(
        image,
        density_pred,
        gt_density_map=gt_density_map,
        alpha=0.8,
        save_path=save_path
    )
def divided_image_patch(inputs, original_w, original_h, crop_size, d_ratio, device):
    """
    分块处理图像并预测密度图
    Args:
        inputs: PIL图像
        original_w: 原始图像宽度
        original_h: 原始图像高度
        crop_size: 裁剪尺寸
        device: 计算设备
    Returns:
        与原始图像尺寸相同的密度图
    """
    # 保存原始尺寸
    orig_w, orig_h = original_w, original_h
    print(f"[DIVIDED] 原始尺寸: {orig_w}x{orig_h}")
    # 1. 调整图像大小
    st_size = 1.0 * min(orig_w, orig_h)

    if st_size < crop_size:
        rr = 1.0 * crop_size / st_size
        wd = int(round(orig_w * rr))
        ht = int(round(orig_h * rr))
        print(f"[DIVIDED] 需要缩放: {orig_w}x{orig_h} -> {wd}x{ht}")
        inputs_resized = inputs.resize((wd, ht), Image.BICUBIC)
    else:
        wd, ht = orig_w, orig_h
        inputs_resized = inputs.copy()
        print(f"[DIVIDED] 无需缩放: {wd}x{ht}")

    # 转换为tensor
    inputs_tensor = img_transformer(inputs_resized)
    inputs_tensor = inputs_tensor.unsqueeze(0).to(device)

    b, c, h, w = inputs_tensor.size()
    print(f"[DIVIDED] 模型输入尺寸: {b}x{c}x{h}x{w}")

    crop_imgs, crop_masks = [], []
    rh, rw = crop_size, crop_size
    mask = torch.zeros([b, 1, h, w]).to(device)

    crop_coords = []

    for i in range(0, h, rh):
        gis = i
        gie = min(h, i + rh)  # 确保不超过边界
        actual_rh = gie - gis

        for j in range(0, w, rw):
            gjs = j
            gje = min(w, j + rw)
            actual_rw = gje - gjs
            # 裁剪图像块
            crop_img = inputs_tensor[:, :, gis:gie, gjs:gje]
            # 如果块尺寸小于crop_size，填充到crop_size
            if actual_rh < rh or actual_rw < rw:
                # 计算填充
                pad_h = rh - actual_rh
                pad_w = rw - actual_rw
                crop_img = F.pad(crop_img, (0, pad_w, 0, pad_h), mode='constant', value=0)
                print(f"[DIVIDED] 填充块: {actual_rh}x{actual_rw} -> {rh}x{rw}")

            crop_imgs.append(crop_img)
            crop_coords.append((gis, gie, gjs, gje))
            mask[:, :, gis:gie, gjs:gje] += 1

    crop_imgs = torch.cat(crop_imgs, dim=0)
    print(f"[DIVIDED] 总块数: {len(crop_coords)}")

    # 3. 批量预测
    crop_preds = []
    batch_size = 1  # 可以根据GPU内存调整

    with torch.no_grad():
        for i in range(0, len(crop_imgs), batch_size):
            gs, gt = i, min(len(crop_imgs), i + batch_size)
            crop_batch = crop_imgs[gs:gt]
            pred = model(crop_batch)
            if d_ratio != 1:
                while pred.dim() < 4:
                    pred = pred.unsqueeze(dim = 0)
                _,_,h1, w1 = pred.size()
                pred = F.interpolate(
                    pred,
                    size=(h1 * d_ratio, w1 * d_ratio),
                    mode="bilinear",
                    align_corners=True
                )

            # 如果预测块是填充的，裁剪回原始块尺寸
            idx = i
            for batch_idx in range(pred.shape[0]):
                gis, gie, gjs, gje = crop_coords[idx + batch_idx]
                actual_rh = gie - gis
                actual_rw = gje - gjs
                # 裁剪预测结果
                pred_crop = pred[batch_idx, :, :actual_rh, :actual_rw]
                # 如果需要，可以在这里进行插值
                crop_preds.append(pred_crop.unsqueeze(0))

    pred_map = torch.zeros([b, 1, h, w]).to(device)
    idx = 0

    for i in range(0, h, rh):
        gis = i
        gie = min(h, i + rh)
        for j in range(0, w, rw):
            gjs = j
            gje = min(w, j + rw)
            if idx < len(crop_preds):
                # 获取对应的预测块
                crop_pred = crop_preds[idx]
                pred_h, pred_w = crop_pred.shape[-2:]
                # 确保块尺寸匹配
                target_h = gie - gis
                target_w = gje - gjs

                if pred_h != target_h or pred_w != target_w:
                    # 调整预测块尺寸
                    crop_pred = F.interpolate(
                        crop_pred.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=True
                    ).squeeze(0)

                pred_map[:, :, gis:gie, gjs:gje] += crop_pred
                idx += 1

    mask = mask.clamp(min=1)  # 避免除零
    outputs = pred_map / mask
    # 6. 调整回原始图像尺寸
    if (h, w) != (orig_h, orig_w):
        print(f"[DIVIDED] 调整回原始尺寸: {h}x{w} -> {orig_h}x{orig_w}")
        outputs = F.interpolate(
            outputs,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True
        )

    return outputs


def main(args, img_dir, npy_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epoch_res = []

    for imgName in os.listdir(img_dir):
        start_time = time.time()

        img_path = os.path.join(img_dir, imgName)
        img = Image.open(img_path).convert("RGB")
        wd, ht = img.size
        print('width = {}  height = {}'.format(wd, ht))
        if wd >= 1024 or ht >= 1024:
            pred_map = divided_image_patch(
                img, wd, ht, 1024,
                d_ratio=args.d_ratio,
                device = device
            )
        else:
            image = img_transformer(img)
            image = image.unsqueeze(0).to(device)
            pred_map = model(image)
            print('here...')

        gt_density_map = np.load(os.path.join(npy_dir, imgName.replace('jpg', 'npy')))
        save_path = os.path.join(save_dir, imgName.replace('.jpg', '.png'))
        process(
            img_path=img_path,
            density_pred=pred_map,
            gt_density_map=gt_density_map,
            save_path=save_path
        )

        gt_count = np.sum(np.load(os.path.join(npy_dir, imgName.replace('jpg', 'npy'))))
        # gt_count = np.sum(pd.read_csv(os.path.join(gt_root, os.path.splitext(imgName)[0] + '.csv'), sep=',', header=None).values)
        pred_count = pred_map.squeeze(0).squeeze(0).cpu().data.numpy().sum()
        # 这里pred_count除以1000，是因为我们在训练模型的时候，给真实的密度图标签值乘以了1000，所以模型预测的结果需要除以1000
        res = gt_count - pred_count / 1000.0
        epoch_res.append(res)

    epoch_res = np.array(epoch_res)
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    print('mae: {}  mse: {}'.format(mae, mse))

# 使用示例
if __name__ == "__main__":
    img_root = r'/home/ff/myProject/KGT/myProjects/myDataset/shanghai/ShanghaiTech/part_A/DCCUS_final/test_data/imgs'
    npy_root = r'/home/ff/myProject/KGT/myProjects/myDataset/shanghai/ShanghaiTech/part_A/DCCUS_final/test_data/npys'
    save_dir = r'/home/ff/myProject/KGT/myProjects/myProjects/CrowdCLIP/myGLPro/datasets/shb/vis'

    from datasets.utils import transforms as T
    args = get_parser()
    device = 'cpu' if torch.cuda.is_available() else 'cpu'

    model, optim, start_epoch, best_mae, best_mse = create_model(args, device=device)
    model.eval()

    normalizer = T.standard_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_transformer = T.standard_transforms.Compose([
        T.standard_transforms.ToTensor(),
        normalizer
    ])

    main(args, img_root, npy_root, save_dir)
    pass



