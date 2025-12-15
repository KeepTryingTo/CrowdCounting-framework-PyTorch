"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/12/6-10:34
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import supervision as sv
import matplotlib.pyplot as plt

import scipy
from scipy.ndimage.filters import gaussian_filter
from sklearn.neighbors import NearestNeighbors

def generate_k_nearest_kernel_densitymap(image,points):
    '''
    Use k nearest kernel to construct the ground truth density map
    for ShanghaiTech PartA.
    image: the image with type numpy.ndarray and [height,width,channel].
    points: the points corresponding to heads with order [col,row].
    '''
    # the height and width of the image
    image_h = image.shape[1]
    image_w = image.shape[2]

    # coordinate of heads in the image
    points_coordinate = points
    # quantity of heads in the image
    points_quantity = len(points_coordinate)

    # generate ground truth density map
    densitymap = np.zeros((image_h, image_w))
    if points_quantity == 0:
        return densitymap
    else:
        pts = np.array(list(zip(np.nonzero(points_coordinate)[1],
                                np.nonzero(points_coordinate)[0])))  # np.nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）
        neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree',
                                     leaf_size=1200)  # https://blog.csdn.net/weixin_37804469/article/details/106911125
        neighbors.fit(pts.copy())
        # 计算当前的每一个标注位置到最近4个点的距离
        distances, _ = neighbors.kneighbors()

        for i, pt in enumerate(points_coordinate):
            pt2d = np.zeros((image_h,image_w), dtype=np.float32)
            if int(pt[1])<image_h and int(pt[0])<image_w:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            else:
                continue
            if points_quantity > 3:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            else:
                # sigma = np.average(np.array(points.shape))/2./2. #case: 1 point
                sigma = 15
            densitymap += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        return densitymap

def gaussian_filter_density_fixed(img, points):
    '''
        This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
        points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
        img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
        return:
        density: the density-map we want. Same shape as input image but only has one channel.
        example:
        points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
        img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[1],img.shape[2]]
    #print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    #print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        # sigma = 4 #np.average(np.array(gt.shape))/2./2. #case: 1 point
        # density += gaussian_filter(pt2d, sigma, truncate=7/sigma, mode='constant')
        sigma = 8
        density += gaussian_filter(pt2d, sigma, mode='constant')
    #print ('done.')
    return density

def read_txt(label_path, img_width, img_height, is_xyxy = True, is_normalize = True):
    """
        :param label_path: 图像中对应坐标框.txt文件路径
        :param img_width: 图像的宽度
        :param img_height: 图像的高度
        :param is_xyxy: .txt文件中每一行包含的坐标框格式，比如[x1, y1, x2, y2]或者[center_x, center_y, width, height]
        :param is_normalize: 坐标框的大小是否被归一化，也是是否被缩放到[0, 1]之间的值，如果是[0, 1]之间的值就需要缩放至相对原图大小
        :return:
    """
    annotations = []
    # try:
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(' ')
        if is_xyxy:
            x1 = float(parts[0])
            y1 = float(parts[1])
            x2 = float(parts[2])
            y2 = float(parts[3])
            width = x2 - x1
            height = y2 - y1
            if is_normalize:
                x1 = x1 * width
                y1 = y1 * height
                center_x = x1 + width / 2
                center_y = y1 + height / 2
            else:
                center_x = x1 + width / 2
                center_y = y1 + height / 2
        else:
            center_x = float(parts[0])
            center_y = float(parts[1])
            width = float(parts[2])
            height = float(parts[3])
            if is_normalize:
                center_x = center_x * width
                center_y = center_y * height

        annotations.append([center_x, center_y])

    # except Exception as e:
    #     print(f"YOLO标注解析错误 {label_path}: {e}")

    return annotations


if __name__ == '__main__':

    # 给定图像的根目录以及box坐标框的目录
    img_root = r'./dataset/data02/test/images'
    label_root = r'./dataset/data02/test/labels'

    # 保存图像和密度图的目录
    save_dir = r'./datasets/peppers/test'
    save_imgs_dir = os.path.join(save_dir, 'imgs')
    save_npys_dir = os.path.join(save_dir, 'npys')

    if not os.path.exists(save_imgs_dir):
        os.makedirs(save_imgs_dir, exist_ok=True)
        print(f'create dir {save_imgs_dir} successed!')

    if not os.path.exists(save_npys_dir):
        os.makedirs(save_npys_dir, exist_ok=True)
        print(f'create dir {save_npys_dir} successed!')

    for imgName in os.listdir(img_root):
        img_path = os.path.join(img_root, imgName)
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        img_height, img_width = img.shape[:2]

        label_name = imgName.split('.')[0] + '.txt'
        label_path = os.path.join(label_root, label_name)
        points = read_txt(label_path, img_width, img_height, is_xyxy=True, is_normalize=True)

        img_zero = np.zeros(shape=(3, img_height, img_width))

        # 高斯平滑生成密度图
        density = gaussian_filter_density_fixed(img_zero, points = points)

        # 保存密度图
        save_npy_path = os.path.join(save_npys_dir, imgName.replace('.jpg', '.npy'))
        np.save(save_npy_path, density)

        # 保存图像
        save_img_path = os.path.join(save_imgs_dir, imgName)
        cv2.imwrite(save_img_path, img)

        print(f'points size: {len(points)}  {imgName} is finished')
    pass