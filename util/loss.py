import torch

import torch.nn as nn

from torch.nn.modules.loss import _Loss


# dice loss=1-dice coefficient

# dice的另一种计算方式：这里参考肾脏肿瘤挑战赛提供的dice计算方法。
def diceCoeffv2(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return score.sum() / N

# v2的另一种计算方法
def diceCoeffv3(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
    # 转为float，以防long类型之间相除结果为0
    score = (2 * tp + eps).float() / (2 * tp + fp + fn + eps).float()

    return score.sum() / N

# 适合用于二分类的Dice loss，最后输出的特征图通道类似[N, 1, H, W]
class BinarySoftDiceLoss(_Loss):

    def __init__(self):
        super(BinarySoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        mean_dice = diceCoeffv2(y_pred, y_true)
        return 1 - mean_dice

# 适合用于多分类的Dice loss，最后输出的特征图通道类似[N, C, H, W]，其中C>2
class SoftDiceLoss(_Loss):

    def __init__(self, num_classes):
        super(SoftDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeffv2(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

class SoftDiceLossV2(_Loss):
    def __init__(self, num_classes, weight=[0.73, 0.73, 0.69, 0.93, 0.92], reduction="sum"):
        super(SoftDiceLossV2, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_pred, y_true):
        class_loss = []
        for i in range(1, self.num_classes):
            dice = diceCoeffv2(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :])
            class_loss.append((1-dice) * self.weight[i-1])
        if self.reduction == 'mean':
            return sum(class_loss) / len(class_loss)
        elif self.reduction == 'sum':
            return sum(class_loss)
        else:
            raise NotImplementedError("no such reduction.")


class WBCELoss(_Loss):
    def __init__(self, num_classes,  smooth=0, size=None, weight=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), reduction='mean', ignore_index=255):
        super(WBCELoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weights = None
        if weight:
            weights = []
            w = torch.ones([1, size, size])
            for v in weight:
                weights.append(w * v)
            self.weights = torch.cat(weights, dim=0)
        self.bce_loss = nn.BCELoss(self.weights, reduction, ignore_index)

    def forward(self, inputs, targets):

        return self.bce_loss(inputs, targets * (1 - self.smooth) + self.smooth / self.num_classes)


class BCE_Dice_Loss(_Loss):
    def __init__(self, num_classes, smooth=0, weight=[1.0, 1.0]):
        super(BCE_Dice_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = SoftDiceLoss(num_classes=num_classes)
        self.weight = weight
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        return self.weight[0] * self.bce_loss(inputs, targets * (1 - self.smooth) + self.smooth / self.num_classes) + self.weight[1] * self.dice_loss(inputs, targets)


class WBCE_Dice_Loss(_Loss):
    def __init__(self, num_classes, smooth=0, size=None, weight=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
        super(WBCE_Dice_Loss, self).__init__()
        self.wbce_loss = WBCELoss(num_classes=num_classes, smooth=smooth, size=size, weight=weight)
        self.dice_loss = SoftDiceLoss(num_classes=num_classes)

    def forward(self, inputs, targets):
        return self.wbce_loss(inputs, targets) + self.dice_loss(inputs, targets)


