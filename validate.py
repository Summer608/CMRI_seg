
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import torch
import numpy as np
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from hausdorff import hausdorff_distance
from segment_anything import sam_model_registry

import sys
parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)  
#personal packages
import util.img_trans as image_transforms
from util import onehot
from dataset import dataloader
from util.loss import *
from util.metric import jaccardv2



model_type = 'sam' #可替换为其他net
fold = 4  # 1 2 3 4 5
augdata = "T2"   #‘DE'，’C0‘, 'T2'

if model_type == 'unet':
    from models.UNet import network
elif model_type == 'transunet':
    from models.Transunet import network
elif model_type == 'sam':
    from models.sam_lora_image_encoder import network



root_path = '/home/caoxiatian/pycharm_project/pj_seg'
val_path = os.path.join(root_path, 'data/MSCMR2019', augdata, 'npy')

input_transform = image_transforms.NpyToTensor()
center_crop = image_transforms.CenterCrop(160)
target_transform = image_transforms.MaskToTensor()

val_set = dataloader.dataloader(val_path, 'val', fold,
                                  joint_transform=None, transform=input_transform, roi_crop=None,
                                  center_crop=center_crop,
                                  target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

palette = dataloader.palette
custom_palette = dataloader.custom_palette
num_classes = dataloader.num_classes

chks = os.listdir(os.path.join(root_path, "checkpoint"))
for i in chks:
    if "{}_fold{}_dice_{}".format(model_type, fold, augdata) in i:
        chk = i
        break
    

sam, img_embedding_size = sam_model_registry["vit_b"](image_size=160,
                                                          num_classes=3,
                                                          checkpoint="/home/caoxiatian/pycharm_project/pj_seg/models/sam_vit_b_01ec64.pth", pixel_mean=[0, 0, 0],
                                                          pixel_std=[1, 1, 1])
net = network(sam, 4).cuda()

#net = network(num_classes=num_classes).cuda()

net.load_state_dict(torch.load(os.path.join(root_path, f"checkpoint/{chk}")))
net.eval()


def auto_val(net):
    # 所有切片的各个类别指标总和
    class_dices = np.array([0] * (num_classes - 1), dtype=np.float)
    class_jaccards = np.array([0] * (num_classes - 1), dtype=np.float)
    class_hsdfs = np.array([0] * (num_classes - 1), dtype=np.float)

    save_path = f'./results/{model_type}_fold{fold}_dice_{augdata}'
    if os.path.exists(save_path):
        # 若该目录已存在，则先删除，用来清空数据
        shutil.rmtree(os.path.join(save_path))
    img_path = os.path.join(save_path, 'images')
    pred_path = os.path.join(save_path, 'pred')
    color_pred_path = os.path.join(save_path, 'color_pred')
    gt_path = os.path.join(save_path, 'gt')
    os.makedirs(img_path)
    os.makedirs(pred_path)
    os.makedirs(color_pred_path)
    os.makedirs(gt_path)

#输出运行日志（比print线程稳定）
    import logging
    import datetime
    log_file = os.path.join(save_path,'logging.log')
    
    logging.basicConfig(level=logging.INFO,
            # format='%(asctime)s %(levelname)s %(message)s',
            format='',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='a')

    logger = logging.getLogger()
    console = logging.StreamHandler()
    logger.addHandler(console)
    logger.info(f'----{datetime.datetime.now()}----')

    # 存放每个切片的指标数组
    val_dice_arr = []
    val_jaccard_arr = []
    val_hsdf_arr = []
    for slice, (input, mask,  file_name) in tqdm(enumerate(val_loader, 1)):
        file_name = file_name[0].split('/')[-1].split('.')[0]

        X = input.cuda()
        # pred = net(X)
        # pred = torch.sigmoid(pred)
        pred = net(X, True, 160)
        pred = torch.sigmoid(pred['masks'])
        pred = pred.cpu().detach()

        # pred[pred < 0.5] = 0
        # pred[pred > 0.5] = 1

        # 原图
        m1 = np.array(input.squeeze())
        m1 = onehot.array_to_img(np.expand_dims(m1, 2))

        # gt
        gt = onehot.onehot_to_mask(np.array(mask.squeeze()).transpose([1, 2, 0]), palette)
        gt = onehot.array_to_img(gt)

        # pred
        save_pred = onehot.onehot_to_mask(np.array(pred.squeeze()).transpose([1, 2, 0]), palette)
        save_pred_png = onehot.array_to_img(save_pred)
        save_pred_color = onehot.onehot_to_mask(np.array(pred.squeeze()).transpose([1, 2, 0]), custom_palette)
        save_pred_png_color = onehot.array_to_img(save_pred_color)

        # 保存预测结果
        # png格式
        m1.save(os.path.join(img_path, file_name + '.png'))
        gt.save(os.path.join(gt_path, file_name + '.png'))
        save_pred_png.save(os.path.join(pred_path, file_name + '.png'))
        save_pred_png_color.save(os.path.join(color_pred_path, file_name + '.png'))

        class_dice = []
        class_jaccard = []
        class_hsdf = []
        for i in range(1, num_classes):
            class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            class_jaccard.append(jaccardv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))
            class_hsdf.append(hausdorff_distance(np.array(pred[0, i:i + 1, :].squeeze()), np.array(mask[0, i:i + 1, :].squeeze()), distance='manhattan'))

        # dice 指标
        val_dice_arr.append(class_dice) # 所有切片的各类别dice [[0.9, 0.9, 0.8], [0.7, 0.8, 0.6], ...]
        class_dices += np.array(class_dice) # 所有切片的各类别dice总和 # eg. [72.29878073 83.99182165 62.14122799]
        # jaccard 指标
        val_jaccard_arr.append(class_jaccard)
        class_jaccards += np.array(class_jaccard)
        # hsdf 指标
        val_hsdf_arr.append(class_hsdf)
        class_hsdfs += np.array(class_hsdf)
        # print('{}, mean_dice: {:.4} - dice_lvm: {:.4} - dice_lv: {:.4} - dice_rv: {:.4}'
        #       .format(slice, sum(class_dice) / 3,  class_dice[0], class_dice[1], class_dice[2]))
    # dice 指标
    dice_slice_arr = np.array(val_dice_arr)
    dice_slice_allclass_mean_arr = np.mean(dice_slice_arr, axis=1)  # 存放所有切片的类别平均dice
    dice_slice_class_mean_arr = np.mean(dice_slice_arr, axis=0)  # 存放所有切片的类别dice
    dice_mean = np.mean(dice_slice_allclass_mean_arr)
    dice_slice_mean_std = np.std(dice_slice_allclass_mean_arr)  # 所有切片的类别平均dice的方差
    dice_slice_std_lvm = np.std(dice_slice_arr[:, 0:1].squeeze())
    dice_slice_std_lv = np.std(dice_slice_arr[:, 1:2].squeeze())
    dice_slice_std_rv = np.std(dice_slice_arr[:, 2:3].squeeze())

    # jaccard 指标
    jaccard_slice_arr = np.array(val_jaccard_arr)
    jaccard_slice_allclass_mean_arr = np.mean(jaccard_slice_arr, axis=1)
    jaccard_slice_class_mean_arr = np.mean(jaccard_slice_arr, axis=0)
    jaccard_mean = np.mean(jaccard_slice_allclass_mean_arr)
    jaccard_slice_mean_std = np.std(jaccard_slice_allclass_mean_arr)
    jaccard_slice_std_lvm = np.std(jaccard_slice_arr[:, 0:1].squeeze())
    jaccard_slice_std_lv = np.std(jaccard_slice_arr[:, 1:2].squeeze())
    jaccard_slice_std_rv = np.std(jaccard_slice_arr[:, 2:3].squeeze())

    # hsdf 指标
    hsdf_slice_arr = np.array(val_hsdf_arr)
    hsdf_slice_allclass_mean_arr = np.mean(hsdf_slice_arr, axis=1)  # 存放所有切片的类别平均dice
    hsdf_slice_class_mean_arr = np.mean(hsdf_slice_arr, axis=0)  # 存放所有切片的类别dice
    hsdf_mean = np.mean(hsdf_slice_allclass_mean_arr)
    hsdf_slice_mean_std = np.std(hsdf_slice_allclass_mean_arr)  # 所有切片的类别平均dice的方差
    hsdf_slice_std_lvm = np.std(hsdf_slice_arr[:, 0:1].squeeze())
    hsdf_slice_std_lv = np.std(hsdf_slice_arr[:, 1:2].squeeze())
    hsdf_slice_std_rv = np.std(hsdf_slice_arr[:, 2:3].squeeze())


    print('mean_dice: {:.3}±{:.3} - dice_Myo: {:.3}±{:.3} - dice_LV: {:.3}±{:.3} - dice_RV: {:.3}±{:.3}'
          .format(dice_mean, dice_slice_mean_std,
                  dice_slice_class_mean_arr[0], dice_slice_std_lvm,
                  dice_slice_class_mean_arr[1], dice_slice_std_lv,
                  dice_slice_class_mean_arr[2], dice_slice_std_rv))
    print('mean_jaccard: {:.3}±{:.3} - jaccard_Myo: {:.3}±{:.3} - jaccard_LV: {:.3}±{:.3} - jaccard_RV: {:.3}±{:.3}'
          .format(jaccard_mean, jaccard_slice_mean_std,
                  jaccard_slice_class_mean_arr[0], jaccard_slice_std_lvm,
                  jaccard_slice_class_mean_arr[1], jaccard_slice_std_lv,
                  jaccard_slice_class_mean_arr[2], jaccard_slice_std_rv))
    print('mean_hsdf: {:.3}±{:.3} - hsdf_Myo: {:.3}±{:.3} - hsdf_LV: {:.3}±{:.3} - hsdf_RV: {:.3}±{:.3}'
          .format(hsdf_mean, hsdf_slice_mean_std,
                  hsdf_slice_class_mean_arr[0], hsdf_slice_std_lvm,
                  hsdf_slice_class_mean_arr[1], hsdf_slice_std_lv,
                  hsdf_slice_class_mean_arr[2], hsdf_slice_std_rv))


if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    auto_val(net)

