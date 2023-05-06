import os
import shutil
import nibabel as nib
import imageio
import numpy as np

from tqdm import tqdm
import cv2

import sys
# sys.path.append('..') #提示系统找到util的位置
parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)

from util import onehot


# 根据数据定义颜色表
palette = [[0], [200], [500], [600]]
# 将mask映射为自定义的颜色
custom_palette = [[0, 0, 0], [255, 255, 255], [101, 12, 68], [68, 104, 118]]

# nii转换为npy和png
def nii_to_npy(dataset_dir, save_path, modal_type='DE', to_png=True):
    im_gt_path = ['mscmr_image', 'mscmr_manual']

    # 所有模态数据
    all_modal_im_namelist = os.listdir(os.path.join(dataset_dir, im_gt_path[0]))
    all_modal_gt_namelist = os.listdir(os.path.join(dataset_dir, im_gt_path[1]))

    # 区分每个模态数据
    im_namelist = [item for item in all_modal_im_namelist if modal_type in item]
    gt_namelist = [item for item in all_modal_gt_namelist if modal_type in item]
    im_namelist.sort(key=lambda x: x.split('_')[0])
    gt_namelist.sort(key=lambda x: x.split('_')[0])

    # 按照图片格式分别命名
    if os.path.exists(os.path.join(save_path, modal_type)):
        shutil.rmtree(os.path.join(save_path, modal_type))
    else:
        os.makedirs(os.path.join(save_path, modal_type, 'npy', 'Images'))
        os.makedirs(os.path.join(save_path, modal_type, 'npy', 'Labels'))
        if to_png:
            os.makedirs(os.path.join(save_path, modal_type, 'color_png', 'Images'))
            os.makedirs(os.path.join(save_path, modal_type, 'color_png', 'Labels'))


    # 每个模态下的所有病例
    for n in tqdm(range(len(gt_namelist))):
        im = nib.load(os.path.join(dataset_dir, im_gt_path[0], im_namelist[n])).dataobj
        gt = nib.load(os.path.join(dataset_dir, im_gt_path[1], gt_namelist[n])).dataobj
        h, w, c = gt.shape
        # 每一个病例的所有切片
        for i in range(c):
            mask = gt[:, :, i]

            # 跳过gt全黑图
            if np.sum(mask) == 0:
                continue
            npy = im[:, :, i]

            # 保存png格式
            if to_png:
                png = onehot.array_to_img(np.expand_dims(npy, axis=2))
                # mask_png = helpers.array_to_img(np.expand_dims(mask, axis=2))

                mask_png_color = onehot.mask_to_onehot(np.expand_dims(mask, axis=2), palette)
                mask_png_color = onehot.onehot_to_mask(mask_png_color, custom_palette)
                mask_png_color = onehot.array_to_img(mask_png_color)

                file_name = im_namelist[n].split('.')[0] + '{}.png'.format(i)

                png.save(os.path.join(save_path, modal_type, 'color_png', 'Images', file_name))
                # mask_png.save(os.path.join(save_path, 'png', 'Labels', file_name))
                mask_png_color.save(os.path.join(save_path, modal_type, 'color_png', 'Labels', file_name))

            # 保存npy格式
            file_name = im_namelist[n].split('.')[0] + '{}.npy'.format(i)
            np.save(os.path.join(save_path, modal_type, 'npy', 'Images', file_name), npy)
            np.save(os.path.join(save_path, modal_type, 'npy', 'Labels', file_name), mask)

            with open(os.path.join(save_path, modal_type, 'npy', 'all.txt'), 'a') as f:
                f.write(file_name)
                f.write('\n')



