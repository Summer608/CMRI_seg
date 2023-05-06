import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
import copy
import cv2


'''
0: background
200: Myo
500: LV
600: RV
'''
palette = [[0], [200], [500], [600]]
# 将mask映射为自定义的颜色
custom_palette = [[0, 0, 0], [255, 255, 255], [101, 12, 68], [68, 104, 118]]
# 分割类别
num_classes = 4

mean_std = ((398.816, 395.903), (242.600, 158.449), (164.044, 182.646))

def get_mean_std(train_loader):    # 方便之后数据标准化处理
    samples, mean, std = 0, 0, 0
    for (input, mask, mask_copy) in tqdm(train_loader):
        samples += 1
        mean += np.mean(input.numpy(), axis=(0, 2, 3))
        std += np.std(input.numpy(), axis=(0, 2, 3))
    mean /= samples
    std /= samples
    print(mean, std)  # 所有批次数据的平均值和标准差

#mask_to_onehot用来将标签进行one-hot
def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


class dataloader(Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, center_crop=None, roi_crop=None, transform=None, target_transform=None):
        # 数据集train/valid/test
        self.imgs = []

        if mode == 'train':
            img_path = os.path.join(root, 'Images')
            mask_path = os.path.join(root, 'Labels')

            if 'Augdata' in root:
                data_list = os.listdir(os.path.join(root, 'Labels'))
            else:
                data_list = [l.strip('\n') for l in open(os.path.join(root, 'train{}.txt'.format(fold))).readlines()]
            for i in data_list:
                self.imgs.append((os.path.join(img_path, i), os.path.join(mask_path, i)))
        elif mode == 'val':
            img_path = os.path.join(root, 'Images')
            mask_path = os.path.join(root, 'Labels')

            data_list = [l.strip('\n') for l in open(os.path.join(root, 'val{}.txt'.format(fold))).readlines()]
            for i in data_list:
                self.imgs.append((os.path.join(img_path, i), os.path.join(mask_path, i)))
        else:
            img_path = os.path.join(root, 'Images')

            data_list = [l.strip('\n') for l in open(os.path.join(root, 'test.txt')).readlines()]
            for i in data_list:
                self.imgs.append(os.path.join(img_path, i))

        self.mode = mode
        self.palette = palette
        self.joint_transform = joint_transform
        self.center_crop = center_crop
        self.roi_crop = roi_crop
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        # 记录原始尺寸
        init_size = 0
        if self.mode is not 'test':
            img_path, mask_path = self.imgs[index]
            file_name = mask_path.split('\\')[-1]

            img = np.load(img_path)
            mask = np.load(mask_path)
            init_size = mask.shape

            img = Image.fromarray(img)
            mask = Image.fromarray(mask)

            if self.joint_transform is not None:
                img, mask = self.joint_transform(img, mask)

            if self.center_crop is not None:
                img, mask = self.center_crop(img, mask)

            img = np.array(img)
            img = np.expand_dims(img, axis=2)
            img = img.transpose([2, 0, 1])

            if self.transform is not None:
                img = self.transform(img)
                # Z-Score
                img = (img - mean_std[0][0]) / mean_std[0][1]

            # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
            # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
            mask = np.expand_dims(mask, axis=2)
            mask = mask_to_onehot(mask, self.palette)

            # shape from (H, W, C) to (C, H, W)
            mask = mask.transpose([2, 0, 1])

            if self.target_transform is not None:
                mask = self.target_transform(mask)

            return img, mask, file_name
        else:
            img_path = self.imgs[index]
            file_name = img_path[0].split('\\')[-1]
            img = np.load(img_path)
            init_size = img.shape
            img = Image.fromarray(img)

            if self.joint_transform is not None:
                img = self.joint_transform(img)
            if self.center_crop is not None:
                img = self.center_crop(img)

            img = np.array(img)
            img = np.expand_dims(img, axis=2)
            img = img.transpose([2, 0, 1])

            if self.transform is not None:
                img = self.transform(img)
                # Z-Score
                img = (img - mean_std[0][0]) / mean_std[0][1]

            return img, file_name, init_size



if __name__ == '__main__':
    np.set_printoptions(threshold=9999999) #控制输出的值的个数为*，其余以...代替

    from torch.utils.data import DataLoader

    import util.img_trans as joint_trans
    from util import onehot

    train_path = r'D:\PycharmProjects\pj_seg\data\MSCMR2019\DE\npy'

    center_crop = joint_trans.CenterCrop(240)
    tes_center_crop = joint_trans.SingleCenterCrop(240)
    train_input_transform = joint_trans.NpyToTensor()

    target_transform = joint_trans.MaskToTensor()

    train_set = dataloader(train_path, 'train', 1, joint_transform=None, center_crop=center_crop, roi_crop=None, transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    for input, mask, _ in train_loader:
        print(input.shape, mask.shape)
        a = copy.copy(mask).squeeze()
        b = onehot.onehot_to_mask(np.array(mask.squeeze()).transpose([1, 2, 0]), custom_palette)
        c = input.squeeze() * mean_std[0][1] + mean_std[0][0]

        cv2.imwrite('mat1.png', np.uint8(np.array(b)))

        cv2.imwrite('mat3.png', np.array(c))




