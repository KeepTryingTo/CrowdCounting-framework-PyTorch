"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/15-13:37
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
from pathlib import Path

def batch_rename_files(directory, old_pattern, new_pattern):
    """批量重命名文件"""
    for filename in os.listdir(directory):
        if old_pattern in filename:
            new_filename = filename.split('.')[0].replace(old_pattern, new_pattern)
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"✅ 重命名: {filename} → {new_filename}")

def batch_rename_files_(directory):
    """批量重命名文件"""
    for mod in ['train', 'test']:
        img_root = os.path.join(directory, mod, 'images')
        lab_root = os.path.join(directory, mod, 'labels')
        img_lists = os.listdir(img_root)
        for i, filename in enumerate(img_lists):
            base_name = filename.split('.')[0].split('_')[1]
            new_img_filename = filename.replace(base_name, str(i))

            old_img_path = os.path.join(img_root, filename)
            new_img_path = os.path.join(img_root, new_img_filename)
            os.rename(old_img_path, new_img_path)
            print(f"✅ 重命名: {filename} → {new_img_filename}")

            new_lab_filename = filename.replace(base_name, str(i)).replace('.jpg', '.txt')
            old_lab_path = os.path.join(lab_root, filename.replace('.jpg' ,'.txt'))
            new_lab_path = os.path.join(lab_root, new_lab_filename)
            os.rename(old_lab_path, new_lab_path)
            print(f"✅ 重命名: {filename.replace('.jpg', '.txt')} → {new_lab_filename}")

if __name__ == '__main__':
    # 使用示例
    root = r'/home/ff/myProject/KGT/myProjects/myProjects/zxCodes/localPeppers/dataset/data02'
    batch_rename_files_(root)
