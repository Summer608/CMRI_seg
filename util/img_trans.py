'''
数据变换集合 包含图像变换、不同形式的变量之间的转化
可用于data augmentation
基于torchvision.transforms
'''


import cv2
import math
import sys
import numbers
import random
from PIL import Image, ImageOps
import torch
import numpy as np

#图像旋转
def data_rotate(im, angle):
    M = cv2.getRotationMatrix2D((im.shape[1] // 2, im.shape[0] // 2), angle, 1.0)
    im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_NEAREST)
    return im

#image transformation

#图像随机缩放函数
class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size=0, scale_rate=0.95, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_rate = scale_rate
        self.fill = fill

    def __call__(self, im, gt):
        img = im.copy()
        mask = gt.copy()
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * self.scale_rate), int(self.base_size * self.scale_rate))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        # ramdom crop
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask

#CenterCrop是以输入图的中心点为中心点做指定size的crop操作
'''
一般数据增强不会采用这个，因为当size固定的时候，在相同输入图像的情况下，N次CenterCrop的结果都是一样的。
'''
class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class SingleCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(math.ceil((w - tw) / 2.))
        y1 = int(math.ceil((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

class Compose(object): #管理图像的transforms
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms: #对img mask进行transform 循环
            img, mask = t(img, mask)
        return img, mask

#相比centercrop更常用的randomcrop
'''
crop时的中心点坐标是随机的，并不是输入图像的中心点坐标，因此基本上每次crop生成的图像都是有差异的。
就是通过 i = random.randint(0, h - th)和 j = random.randint(0, w - tw)两行生成一个随机中心点的横纵坐标。
'''
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))
class CenterCrop_npy(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.shape == mask.shape
        if (self.size <= img.shape[1]) and (self.size <= img.shape[0]):
            x = math.ceil((img.shape[1] - self.size) / 2.)
            y = math.ceil((img.shape[0] - self.size) / 2.)

            if len(mask.shape) == 3:
                return img[y:y + self.size, x:x + self.size, :], mask[y:y + self.size, x:x + self.size, :]
            else:
                return img[y:y + self.size, x:x + self.size, :], mask[y:y + self.size, x:x + self.size]
        else:
            raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
                self.size, self.size, img.shape[0], img.shape[1]))

#随机的图像水平翻转，通俗讲就是图像的左右对调。
class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5: #水平翻转的概率为0.5
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size=0, scale_rate=0.95, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_rate = scale_rate
        self.fill = fill

    def __call__(self, im, gt):
        img = im.copy()
        mask = gt.copy()
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * self.scale_rate), int(self.base_size * self.scale_rate))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

#transform


class NpyToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.float32))

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.float32))

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):  #输入必须为tensor
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# 不带归一化
class ImgToTensor(object):
    def __call__(self, img):
        img = torch.from_numpy(np.array(img))
        if isinstance(img, torch.ByteTensor):
            return img.float()