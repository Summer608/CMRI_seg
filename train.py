import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import time
import random
from torch.utils.data import DataLoader
import util.img_trans as joint_trans

from segment_anything import sam_model_registry

import sys
parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)

from dataset import dataloader
from util.loss import *
from util.loss import diceCoeffv2
from util.pytorchtool import EarlyStopping
from util import misc


# 超参数
num_classes = 4
crop_size = 160
batch_size = 32  # unet 36 trans 32
n_epoch = 100
lr_scheduler_eps = 1e-3
lr_scheduler_patience = 10
early_stop_patience = 12
initial_lr = 1e-4    # 1e-4
threshold_lr = 1e-6
weight_decay = 1e-5  # 1e-5
optimizer_type = 'adam'
scheduler_type = 'no'
label_smoothing = 0.01
aux_loss = False
gamma = 0.5
alpha = 0.85
model_number = random.randint(1, 1e6)


model_type = 'sam'

if model_type == 'unet':
    from models.UNet import network
elif model_type == 'transunet':
    from models.Transunet import network
elif model_type == 'sam':
    from models.sam_lora_image_encoder import network


def main(network, fold, loss_name, train_path, val_path):
    # net = network(num_classes=num_classes).cuda()
    sam, img_embedding_size = sam_model_registry["vit_b"](image_size=160,
                                                          num_classes=3,
                                                          checkpoint="/home/caoxiatian/pycharm_project/pj_seg/models/sam_vit_b_01ec64.pth", pixel_mean=[0, 0, 0],
                                                          pixel_std=[1, 1, 1])
    net = network(sam, 4).cuda()

    # 数据预处理，加载
    center_crop = joint_trans.CenterCrop(crop_size)
    input_transform = joint_trans.NpyToTensor()
    target_transform = joint_trans.MaskToTensor()

    train_set = dataloader.dataloader(train_path, 'train', fold, joint_transform=None, roi_crop=None,
                                    center_crop=center_crop,
                                    transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=batch_size)
    val_set = dataloader.dataloader(val_path, 'val', fold,
                                  joint_transform=None, transform=input_transform, roi_crop=None,
                                  center_crop=center_crop,
                                  target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 定义损失函数
    if loss_name == 'dice':
        criterion = SoftDiceLoss(num_classes).cuda()
    if loss_name =='bce':
        criterion = BCE_Dice_Loss(num_classes).cuda()

    # 定义早停机制
    early_stopping = EarlyStopping(early_stop_patience, verbose=True, delta=lr_scheduler_eps,
                                   path=os.path.join(root_path, 'checkpoint', '{}.pth'.format(model_name)))

    # 定义优化器
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    train(train_loader, val_loader, net, criterion, optimizer, early_stopping, n_epoch, 0)


def train(train_loader, val_loader, net, criterion, optimizer, early_stopping, num_epoches, iters):
    for epoch in range(1, num_epoches+1):
        st = time.time()
        train_class_dices = np.array([0] * (dataloader.num_classes - 1), dtype=np.float)
        val_class_dices = np.array([0] * (dataloader.num_classes - 1), dtype=np.float)
        val_dice_arr = []
        train_losses = []
        val_losses = []
        # batch = 1
        # val_batch = 1

        # 训练
        net.train()
        for batch, (input, mask, __) in enumerate(train_loader, 1):
            x1 = input.cuda()
            y = mask.cuda()
            optimizer.zero_grad()
            # output = net(x1)
            output = net(x1, True, 160)
            output = torch.sigmoid(output['masks'])
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            iters += 1

            train_losses.append(loss.item())

            class_dice = []
            for i in range(1, num_classes):
                our_dice = diceCoeffv2(output[:, i:i+1, :], y[:, i:i+1, :]).cpu().item()
                class_dice.append(our_dice)

            mean_dice = sum(class_dice)/len(class_dice)
            train_class_dices += np.array(class_dice)

            string_print = 'epoch: {} - iters: {} - loss: {:.4} - mean: {:.4} - Myo: {:.4}- LV: {:.4} - RV: {:.4} - time: {:.2}' \
                .format(epoch, iters, loss.data.cpu(), mean_dice, class_dice[0], class_dice[1], class_dice[2], time.time() - st)
            misc.log(string_print)

            st = time.time()

        train_loss = np.average(train_losses)
        train_class_dices = train_class_dices / batch
        train_mean_dice = train_class_dices.sum() / train_class_dices.size


        print('epoch {}/{} - train_loss: {:.4} - train_mean_dice: {:.4} - dice_Myo: {:.4} - dice_LV: {:.4} - dice_RV: {:.4}'.format(
                epoch, num_epoches, train_loss, train_mean_dice, train_class_dices[0], train_class_dices[1], train_class_dices[2]))

        # 验证
        net.eval()
        for val_batch, (input, mask, __) in enumerate(val_loader, 1):
            val_x1 = input.cuda()
            val_y = mask.cuda()
            #pred = net(val_x1)
            pred = net(val_x1, True, 160)
            pred = torch.sigmoid(pred['masks'])
            val_loss = criterion(pred, val_y)
            val_losses.append(val_loss.item())
            pred = pred.cpu().detach()

            val_class_dice = []
            for i in range(1, num_classes):
                val_class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))

            val_dice_arr.append(val_class_dice)
            val_class_dices += np.array(val_class_dice)

        val_loss = np.average(val_losses)

        val_dice_arr = np.array(val_dice_arr)
        std = (np.std(val_dice_arr[:, 1:2]) + np.std(val_dice_arr[:, 2:3]) + np.std(
            val_dice_arr[:, 3:4])) / num_classes
        val_class_dices = val_class_dices / val_batch
        val_mean_dice = val_class_dices.sum() / val_class_dices.size

        organ_mean_dice = (val_class_dices[0] + val_class_dices[1] + val_class_dices[2]) / num_classes


        print('val_loss: {:.4} - val_mean_dice: {:.4} - mean: {:.4}±{:.3} - Myo: {:.4}- LV: {:.4} - RV: {:.4}'
              .format(val_loss, val_mean_dice, organ_mean_dice, std, val_class_dices[0], val_class_dices[1], val_class_dices[2]))
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        early_stopping(organ_mean_dice, net, epoch)
        if early_stopping.early_stop or optimizer.param_groups[0]['lr'] < threshold_lr:
            print("Early stopping")
            # 结束模型训练
            break

    print('----------------------------------------------------------')
    print('save epoch {}'.format(early_stopping.save_epoch))
    print('stoped epoch {}'.format(epoch))
    print('----------------------------------------------------------')






if __name__ == '__main__':
    # 5折交叉验证训练模型
    augdata = ["DE", "C0", "T2"]
    folds = [1, 2, 3, 4, 5]
    dataset_name = 'MSCMR2019'



    root_path = '/home/caoxiatian/pycharm_project/pj_seg'

    for modal in augdata:
        for fold in folds:
            loss_name = 'dice'  # 可替换的损失函数：dice, bce, wbce, dual, wdual
            reduction = '' + modal
            model_name = '{}_fold{}_{}_{}_{}'.format(model_type, fold, loss_name, reduction, model_number)

            train_path = os.path.join(root_path, 'data/{}/Augdata{}'.format(dataset_name, modal), 'train{}'.format(fold), 'npy')
            val_path = os.path.join(root_path, 'data/{}'.format(dataset_name), modal, 'npy')

            main(network=network, fold=fold, loss_name=loss_name, train_path=train_path, val_path=val_path)


