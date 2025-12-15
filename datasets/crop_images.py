import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


class AdvancedYOLOCropper:
    def __init__(self, images_dir: str, labels_dir: str,
                 save_images_dir: str, save_labels_dir: str,
                 save_labels_proc_res: str,
                 target_class: int = 1,
                 other_class: int = 0,
                 padding: int = 0):
        """
        增强版YOLO裁剪器 - 同时处理图像和坐标转换

        Args:
            images_dir: 原始图像目录
            labels_dir: 原始标签目录
            save_images_dir: 裁剪图像保存目录
            save_labels_dir: 新标签保存目录
            target_class: 要裁剪的目标类别（默认1）
            other_class: 要保存坐标的其他类别（默认0）
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.save_images_dir = Path(save_images_dir)
        self.save_labels_dir = Path(save_labels_dir)
        self.save_labels_proc_res = Path(save_labels_proc_res)
        self.target_class = target_class
        self.other_class = other_class
        self.padding = padding

        # 创建保存目录
        self.save_images_dir.mkdir(parents=True, exist_ok=True)
        self.save_labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"裁剪目标: 类别 {target_class}")
        print(f"保存坐标: 类别 {other_class}")
        print(f"图像保存: {save_images_dir}")
        print(f"标签保存: {save_labels_dir}")
        print("=" * 50)

    def convert_coordinates(self, original_coords: List[float],
                            crop_region: Tuple[int, int, int, int],
                            original_size: Tuple[int, int]) -> List[float]:
        """
        转换坐标到裁剪后的相对坐标系

        Args:
            original_coords: 原始坐标 [class, x_center, y_center, width, height]
            crop_region: 裁剪区域 (x1, y1, x2, y2)
            original_size: 原始图像尺寸 (width, height)

        Returns:
            List[float]: 转换后的坐标 [class, new_x_center, new_y_center, new_width, new_height]
        """
        x1, y1, x2, y2 = crop_region
        orig_w, orig_h = original_size

        # 裁剪区域的尺寸
        crop_w = x2 - x1
        crop_h = y2 - y1

        # 原始坐标（归一化→像素坐标）
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = original_coords
        x_center_px = x_center_norm * orig_w
        y_center_px = y_center_norm * orig_h
        width_px = width_norm * orig_w
        height_px = height_norm * orig_h

        # 计算原始边界框
        box_x1 = x_center_px - width_px / 2
        box_y1 = y_center_px - height_px / 2
        box_x2 = x_center_px + width_px / 2
        box_y2 = y_center_px + height_px / 2

        # 检查是否在裁剪区域内
        if (box_x2 < x1 or box_x1 > x2 or
                box_y2 < y1 or box_y1 > y2):
            return None  # 完全在裁剪区域外

        # 计算相交区域
        intersect_x1 = max(box_x1, x1)
        intersect_y1 = max(box_y1, y1)
        intersect_x2 = min(box_x2, x2)
        intersect_y2 = min(box_y2, y2)

        # 计算新的中心点和尺寸（相对于裁剪区域）
        new_x_center = ((intersect_x1 + intersect_x2) / 2 - x1) / crop_w
        new_y_center = ((intersect_y1 + intersect_y2) / 2 - y1) / crop_h
        new_width = (intersect_x2 - intersect_x1) / crop_w
        new_height = (intersect_y2 - intersect_y1) / crop_h

        # 确保坐标在[0,1]范围内
        new_x_center = max(0.0, min(1.0, new_x_center))
        new_y_center = max(0.0, min(1.0, new_y_center))
        new_width = max(0.0, min(1.0, new_width))
        new_height = max(0.0, min(1.0, new_height))

        # 忽略太小的目标
        if new_width < 0.01 or new_height < 0.01:
            return None

        return [class_id, new_x_center, new_y_center, new_width, new_height]

    def process_single_image(self, image_path: Path, padding: int = 1) -> Dict:
        """
        处理单张图像：裁剪目标+转换坐标

        Returns:
            Dict: 处理结果统计
        """
        image_name = image_path.stem
        label_path = self.labels_dir / f"{image_name}.txt"

        if not label_path.exists():
            return {'success': False, 'reason': '标签文件不存在'}

        # 读取图像和标签
        img = cv2.imread(str(image_path))
        if img is None:
            return {'success': False, 'reason': '无法读取图像'}

        orig_h, orig_w = img.shape[:2]
        original_size = (orig_w, orig_h)

        # 读取所有标签
        all_annotations = self._read_yolo_annotations(label_path)

        # 分离目标类别和其他类别
        target_annotations = [ann for ann in all_annotations if ann[0] == self.target_class]
        other_annotations = [ann for ann in all_annotations if ann[0] == self.other_class]

        results = {
            'image_name': image_name,
            'target_crops': [],
            'converted_labels': []
        }

        # 处理每个目标裁剪区域
        for i, target_ann in enumerate(target_annotations):
            # 获取裁剪区域坐标
            crop_region = self._get_crop_region(target_ann, original_size, padding)
            x1, y1, x2, y2 = crop_region

            # 裁剪图像
            cropped_img = img[y1:y2, x1:x2]
            if cropped_img.size == 0:
                continue

            # 保存裁剪图像
            crop_img_name = f"{image_name}_crop{i + 1}.jpg"
            crop_img_path = self.save_images_dir / crop_img_name
            cv2.imwrite(str(crop_img_path), cropped_img)

            # 转换其他类别的坐标
            converted_coords = []
            for other_ann in other_annotations:
                new_coords = self.convert_coordinates(other_ann, crop_region, original_size)
                if new_coords:
                    converted_coords.append(new_coords)

            # 保存转换后的标签
            if converted_coords:
                label_filename = f"{image_name}_crop{i + 1}.txt"
                label_path = self.save_labels_dir / label_filename
                self._save_yolo_annotations(label_path, converted_coords)

                results['converted_labels'].append({
                    'label_file': label_filename,
                    'coord_count': len(converted_coords)
                })

            results['target_crops'].append({
                'crop_image': crop_img_name,
                'crop_region': crop_region,
                'converted_labels_count': len(converted_coords)
            })

        return results

    def _get_crop_region(self, annotation: List[float],
                         original_size: Tuple[int, int],
                         padding: int) -> Tuple[int, int, int, int]:
        """计算裁剪区域坐标"""
        orig_w, orig_h = original_size
        class_id, x_center, y_center, width, height = annotation

        # 转换为相对于原图大小的坐标
        x_center_px = x_center * orig_w
        y_center_px = y_center * orig_h
        width_px = width * orig_w
        height_px = height * orig_h

        # 计算边界框
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)

        # 添加填充边界，相当于坐标框往外扩充了pad大小的边界，保存目标一定是在裁剪的范围内
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(orig_w, x2 + padding)
        y2_pad = min(orig_h, y2 + padding)

        return (x1_pad, y1_pad, x2_pad, y2_pad)

    def _read_yolo_annotations(self, label_path: Path) -> List[List[float]]:
        """读取YOLO格式的标签文件"""
        annotations = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        annotation = [float(part) for part in parts[:5]]
                        annotations.append(annotation)
        except:
            pass
        return annotations

    def _save_yolo_annotations(self, save_path: Path, annotations: List[List[float]]):
        """保存YOLO格式的标签文件"""
        with open(save_path, 'w') as f:
            for ann in annotations:
                line = ' '.join(f'{x:.6f}' for x in ann)
                f.write(line + '\n')

    def batch_process(self):
        """批量处理所有图像"""
        image_files = list(self.images_dir.glob('*.*'))
        image_files = [f for f in image_files
                       if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG']]

        print(f"找到 {len(image_files)} 个图像文件")
        print("开始批量处理图像和坐标转换...")
        print("=" * 50)

        statistics = {
            'total_processed': 0,
            'successful_images': 0,
            'total_crops': 0,
            'total_labels_saved': 0,
            'details': []
        }

        for image_path in image_files:
            result = self.process_single_image(image_path, self.padding)
            statistics['total_processed'] += 1

            if result['target_crops']:
                statistics['successful_images'] += 1
                statistics['total_crops'] += len(result['target_crops'])
                statistics['total_labels_saved'] += len(result['converted_labels'])

                print(f"{image_path.name}: "
                      f"裁剪 {len(result['target_crops'])} 个区域, "
                      f"生成 {len(result['converted_labels'])} 个标签文件")

            statistics['details'].append(result)

        # 生成统计报告
        self._generate_statistics_report(statistics)
        return statistics

    def _generate_statistics_report(self, stats: Dict):
        """生成处理统计报告"""
        report_path = self.save_labels_proc_res / "processing_report.txt"

        with open(report_path, 'w') as f:
            f.write("YOLO图像裁剪与坐标转换报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {np.datetime64('now')}\n")
            f.write(f"原始图像: {self.images_dir}\n")
            f.write(f"原始标签: {self.labels_dir}\n")
            f.write(f"裁剪图像: {self.save_images_dir}\n")
            f.write(f"转换标签: {self.save_labels_dir}\n")
            f.write(f"目标类别: {self.target_class}\n")
            f.write(f"保存类别: {self.other_class}\n\n")

            f.write("处理统计:\n")
            f.write(f"总处理图像: {stats['total_processed']}\n")
            f.write(f"成功处理: {stats['successful_images']}\n")
            f.write(f"总裁剪区域: {stats['total_crops']}\n")
            f.write(f"总生成标签: {stats['total_labels_saved']}\n")

            # 详细统计
            f.write("\n详细结果:\n")
            for detail in stats['details']:
                f.write(f"\n图像: {detail['image_name']}\n")
                f.write(f"裁剪区域: {len(detail['target_crops'])}\n")
                f.write(f"生成标签文件: {len(detail['converted_labels'])}\n")
                for label in detail['converted_labels']:
                    f.write(f"  - {label['label_file']}: {label['coord_count']} 个坐标\n")


# 使用示例
def main():
    root = r'/home/ubuntu/Documents/KTG/myProjects/yolov13-main/dataset/PepperDetect/data01'
    save_dir = r'/home/ubuntu/Documents/KTG/myProjects/yolov13-main/dataset/PepperDetect/data02'
    for mod in ['test']:
        img_root = os.path.join(root, mod, 'images')
        lab_root = os.path.join(root, mod, 'labels')
        """主函数 - 使用示例"""
        cropper = AdvancedYOLOCropper(
            images_dir=img_root,
            labels_dir=lab_root,
            save_images_dir=os.path.join(save_dir, mod, 'images'),
            save_labels_dir=os.path.join(save_dir, mod, 'labels'),
            save_labels_proc_res = save_dir,
            target_class=1,  # 裁剪类别1
            other_class=0  # 保存类别0的坐标
        )

        statistics = cropper.batch_process()

        print(f"\n{mod} 处理完成！")
        print(f"{mod} 裁剪图像保存在: {os.path.join(save_dir, mod, 'images')}")
        print(f"{mod} 转换标签保存在: {os.path.join(save_dir, mod, 'labels')}")
        print(f"{mod} 统计报告: ./data02/processing_report.txt")


if __name__ == "__main__":
    main()