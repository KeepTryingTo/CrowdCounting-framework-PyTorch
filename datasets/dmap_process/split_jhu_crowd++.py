"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/4/2-13:06
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import cv2
import shutil
from glob import glob



def process_jhu(root):
    # root = r'/home/ff/myProject/KGT/myProjects/myDataset/jhu_crowd_v2.0'

    for phrase in ['train', 'test', 'val']:
        image_labels = os.path.join(root,phrase, 'image_labels.txt')
        with open(image_labels, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        #TODO 图像根据天气场景来分类(记录的是文件的路径，以便于下面实现文件复制）
        FH_imgs = []
        FH_npys = []
        SN_imgs = []
        SN_npys = []

        SD_imgs = []
        SD_npys = []
        SR_imgs = []
        SR_npys = []

        #TODO 得到要复制的目标图像和密度图的路径
        for line in lines:
            fileName,count,scene_type,weather_cond,distractor = line.split(',')
            #TODO 如果天气类别属于3-雪或者1-雾/霾天气
            if int(weather_cond) == 1:
                FH_imgs.append(os.path.join(root,phrase,'FH','imgs',fileName + '.jpg'))
                FH_npys.append(os.path.join(root,phrase,'FH','npys',fileName + '.npy'))
            elif int(weather_cond) == 3:
                SN_imgs.append(os.path.join(root, phrase, 'SN', 'imgs', fileName + '.jpg'))
                SN_npys.append(os.path.join(root, phrase, 'SN', 'npys', fileName + '.npy'))

            #TODO 如果场景类别属于street-街道和stadium-体育场
            if scene_type == 'street':
                SR_imgs.append(os.path.join(root,phrase,'SR','imgs',fileName + '.jpg'))
                SR_npys.append(os.path.join(root,phrase,'SR','npys',fileName + '.npy'))
            elif scene_type == 'stadium':
                SD_imgs.append(os.path.join(root, phrase, 'SD', 'imgs', fileName + '.jpg'))
                SD_npys.append(os.path.join(root, phrase, 'SD', 'npys', fileName + '.npy'))
            print(f'{fileName} is finished!')
        #TODO 根据路径实现文件复制操作
        #1 首先复制图像
        for img_path in FH_imgs:
            imgName = img_path.split('/')[-1]
            source_img_path = os.path.join(root,phrase,'images',imgName)
            shutil.copy(source_img_path,img_path)
        for img_path in SN_imgs:
            imgName = img_path.split('/')[-1]
            source_img_path = os.path.join(root,phrase,'images',imgName)
            shutil.copy(source_img_path,img_path)

        for img_path in SD_imgs:
            imgName = img_path.split('/')[-1]
            source_img_path = os.path.join(root,phrase,'images',imgName)
            shutil.copy(source_img_path,img_path)

        for img_path in SR_imgs:
            imgName = img_path.split('/')[-1]
            source_img_path = os.path.join(root,phrase,'images',imgName)
            shutil.copy(source_img_path,img_path)
        print(f'{phrase} copy images is done!')

        # 2 其次是复制密度图
        for npy_path in FH_npys:
            npyName = npy_path.split('/')[-1]
            source_img_path = os.path.join(root, phrase, 'npys', npyName)
            shutil.copy(source_img_path, npy_path)
        for npy_path in SN_npys:
            npyName = npy_path.split('/')[-1]
            source_img_path = os.path.join(root, phrase, 'npys', npyName)
            shutil.copy(source_img_path, npy_path)

        for npy_path in SD_npys:
            npyName = npy_path.split('/')[-1]
            source_img_path = os.path.join(root, phrase, 'npys', npyName)
            shutil.copy(source_img_path, npy_path)

        for npy_path in SR_npys:
            npyName = npy_path.split('/')[-1]
            source_img_path = os.path.join(root, phrase, 'npys', npyName)
            shutil.copy(source_img_path, npy_path)

        print(f'{phrase} copy npys is done!')

def process_jhu_domains():
    root_dir = r'/home/ff/myProject/KGT/myProjects/myDataset/jhu_crowd_v2.0'
    save_dir = r'/home/ff/myProject/KGT/myProjects/myDataset/jhu_crowd_v2.0/FH'

    for domainName in ['fog','snow','stadium','street']:
        val_path = os.path.join(
            root_dir,
            'jhu_domains',
            'jhu_' + domainName + '_val.txt',
        )
        train_path = os.path.join(
            root_dir,
            'jhu_domains',
            'jhu_' + domainName + '_train.txt',
        )

        if domainName == 'fog':
            save_dir = os.path.join(
                root_dir,
                'FH'
            )
        elif domainName == 'snow':
            save_dir = os.path.join(
                root_dir,
                'SN'
            )
        elif domainName == 'stadium':
            save_dir = os.path.join(
                root_dir,
                'SD'
            )
        elif domainName == 'street':
            save_dir = os.path.join(
                root_dir,
                'SR'
            )
        # TODO 保存训练集和测试集的图像和密度图路径，用于后面的复制操作
        test_imgs_path = []
        test_npys_path = []
        train_imgs_path = []
        train_npys_path = []
        with open(train_path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        imgName_list = [line.split('/')[-1].strip() for line in lines]

        for phrase in ['train', 'test', 'val']:
            s_img_path = os.path.join(
                root_dir,
                phrase,
                'images'
            )
            #TODO 当前训练集或验证集或测试集的图像
            imgs_list = [img_path.split('/')[-1].strip() for img_path in  glob(os.path.join(s_img_path, '*.jpg'))]
            for imgName in imgName_list:
                if imgName in imgs_list:
                    train_imgs_path.append(
                        os.path.join(s_img_path,imgName)
                    )
                    train_npys_path.append(
                        os.path.join(
                            root_dir,
                            phrase,
                            'npys',
                            imgName.replace('.jpg','.npy')
                        )
                    )
        with open(val_path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        imgName_list = [line.split('/')[-1].strip() for line in lines]

        for phrase in ['train', 'test', 'val']:
            s_img_path = os.path.join(
                root_dir,
                phrase,
                'images'
            )
            # TODO 当前训练集或验证集或测试集的图像
            imgs_list = [img_path.split('/')[-1].strip() for img_path in glob(os.path.join(s_img_path, '*.jpg'))]
            for imgName in imgName_list:
                if imgName in imgs_list:
                    test_imgs_path.append(
                        os.path.join(s_img_path, imgName)
                    )
                    test_npys_path.append(
                        os.path.join(
                            root_dir,
                            phrase,
                            'npys',
                            imgName.replace('.jpg','.npy')
                        )
                    )
        #TODO 复制训练集和测试集到指定保存的目录
        for img_path in train_imgs_path:
            t_img_path = os.path.join(
                save_dir,
                'train_data',
                'imgs',
                img_path.split('/')[-1]
            )
            shutil.copy(img_path,t_img_path)

        for img_path in test_imgs_path:
            t_img_path = os.path.join(
                save_dir,
                'test_data',
                'imgs',
                img_path.split('/')[-1]
            )
            shutil.copy(img_path,t_img_path)

        for npy_path in train_npys_path:
            t_npy_path = os.path.join(
                save_dir,
                'train_data',
                'npys',
                npy_path.split('/')[-1]
            )
            shutil.copy(npy_path,t_npy_path)

        for npy_path in test_npys_path:
            t_npy_path = os.path.join(
                save_dir,
                'test_data',
                'npys',
                npy_path.split('/')[-1]
            )
            shutil.copy(npy_path, t_npy_path)
        print(f'{domainName} is done!')


if __name__ == '__main__':
    # process_jhu()
    process_jhu_domains()
    """
    train:
        SD: 482
        SR: 291
        FH: 81
        SN: 102
    val  : 
        SD: 101
        SR: 66
        FH: 23
        SN: 21
    test: 
        SD: 296
        SR: 215
        FH: 64
        SN: 78
        
    total SD : 879
    total SR : 572
    total FH : 168
    total SN : 201
    """
    pass