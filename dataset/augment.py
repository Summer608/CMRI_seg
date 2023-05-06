import os
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

from util.img_trans import RandomScaleCrop
from util.img_trans import data_rotate
from util import onehot

from niitrans import *


# 数据增广（为后续train作准备）
def data_augmentation(dataset_dir, save_path, modal_type='T2', to_png=True):
    save_path_init = dataset_dir
    aug_data_files = ['train1', 'train2', 'train3', 'train4', 'train5']
    angles = [45, 90, 135, 180, 225, 270, 315]
    scale_rate = [0.75, 0.8, 0.9, 1, 1.1, 1.25]
    flip_codes = [1, 0, -1]
    aug_file_name = 'Augdata{}'.format(modal_type)
    for aug_data_file in aug_data_files:
        items = []
        img_path = os.path.join(dataset_dir, modal_type, 'npy')
        data_list = [l.strip('\n') for l in open(os.path.join(img_path, '{}.txt'.format(aug_data_file))).readlines()]
        # 获取图像路径
        for it in data_list:
            item = (os.path.join(img_path, 'Images', it), os.path.join(img_path, 'Labels', it))
            items.append(item)

        # 创建增广数据目录
        if os.path.exists(os.path.join(save_path_init, aug_file_name, aug_data_file)):
            # 若该目录已存在，则先删除，用来清空数据
            print('清空原始数据中...')
            shutil.rmtree(os.path.join(save_path_init, aug_file_name, aug_data_file))
            print('原始数据已清空。')

        save_path = os.path.join(save_path_init, aug_file_name, aug_data_file)
        npy_save_path = os.path.join(save_path, 'npy')

        os.makedirs(os.path.join(npy_save_path, 'Images'))
        os.makedirs(os.path.join(npy_save_path, 'Labels'))

        if to_png:
            png_save_path = os.path.join(save_path, 'png')
            os.makedirs(os.path.join(png_save_path, 'Images'))
            os.makedirs(os.path.join(png_save_path, 'Labels'))
        # 加载图像
        for item in tqdm(items):
            img_path, mask_path = item
            file_name = mask_path.split('/')[-1][:-4] # patient18_C09
            im = np.load(img_path)
            gt = np.load(mask_path)

            # 旋转
            for angle in angles:
                img_ = data_rotate(im, angle)
                gt_ = data_rotate(gt, angle)
                extra = '_rotate{}'.format(angle)

                if to_png:
                    png_file_name = file_name + '{}.png'.format(extra)
                    png_im = onehot.array_to_img(np.expand_dims(img_, axis=2))
                    png_gt = onehot.array_to_img(np.expand_dims(gt_, axis=2))
                    png_im.save(os.path.join(png_save_path, 'Images', png_file_name))
                    png_gt.save(os.path.join(png_save_path, 'Labels', png_file_name))

                npy_file_name = file_name + '{}.npy'.format(extra)
                np.save(os.path.join(npy_save_path, 'Images', npy_file_name), img_)
                np.save(os.path.join(npy_save_path, 'Labels', npy_file_name), gt_)

                # 三个模态只需保存任意一次名字

                with open(os.path.join(save_path, 'train.txt'), 'a') as f:
                    f.write(npy_file_name)
                    f.write('\n')
            # 随机缩放
            for sr in scale_rate:
                SR = RandomScaleCrop(256, 256, scale_rate=sr)
                extra = '_scale{}'.format(sr)
                img_, gt_ = SR(Image.fromarray(im), Image.fromarray(gt))
                img_, gt_ = np.array(img_), np.array(gt_)

                if to_png:
                    png_file_name = file_name + '{}.png'.format(extra)
                    png_im = onehot.array_to_img(np.expand_dims(img_, axis=2))
                    png_gt = onehot.array_to_img(np.expand_dims(gt_, axis=2))
                    png_im.save(os.path.join(png_save_path, 'Images', png_file_name))
                    png_gt.save(os.path.join(png_save_path, 'Labels', png_file_name))

                npy_file_name = file_name + '{}.npy'.format(extra)
                np.save(os.path.join(npy_save_path, 'Images', npy_file_name), img_)
                np.save(os.path.join(npy_save_path, 'Labels', npy_file_name), gt_)
                with open(os.path.join(save_path, 'train.txt'), 'a') as f:
                    f.write(npy_file_name)
                    f.write('\n')

            # 翻转
            for code in flip_codes:
                img_ = cv2.flip(im, code)
                gt_ = cv2.flip(gt, code)
                if to_png:
                    png_file_name = file_name + '_flip{}.png'.format(code)
                    png_im = onehot.array_to_img(np.expand_dims(img_, axis=2))
                    png_gt = onehot.array_to_img(np.expand_dims(gt_, axis=2))
                    png_im.save(os.path.join(png_save_path, 'Images', png_file_name))
                    png_gt.save(os.path.join(png_save_path, 'Labels', png_file_name))

                npy_file_name = file_name + '_flip{}.npy'.format(code)
                np.save(os.path.join(npy_save_path, 'Images', npy_file_name), img_)
                np.save(os.path.join(npy_save_path, 'Labels', npy_file_name), gt_)

                with open(os.path.join(save_path, 'train.txt'), 'a') as f:
                    f.write(npy_file_name)
                    f.write('\n')


if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    # nii转换为2d切片
    types = ['DE', 'C0', 'T2']
    #nii_to_npy(r'D:\PycharmProjects\pj_seg\data', r'D:\PycharmProjects\pj_seg\data\MSCMR2019', modal_type=types[2])


    data_augmentation(r'D:\PycharmProjects\pj_seg\data\MSCMR2019', r'D:\PycharmProjects\pj_seg\data', modal_type=types[0])

