import torch
from torch import nn
from torch.nn import functional as F


# 双层卷积
class Doubleconv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(Doubleconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

# 下采样模块
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mp = nn.MaxPool2d(2, stride=2)
        self.conv = Doubleconv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(self.mp(x))
        return x

# 上采样模块
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Doubleconv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2]-x1.size()[2]
        diff_x = x2.size()[3]-x1.size()[3]
        x1 = F.pad(x1, [diff_x//2, diff_x-diff_x//2,
                       diff_y//2, diff_y-diff_y//2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# 网络模型
class network(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(network, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.in_conv = Doubleconv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 128)

        self.up1 = Up(256, 64)
        self.up2 = Up(128, 32)
        self.up3 = Up(64, 16)
        self.up4 = Up(32, 16)

        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)


    def forward(self, x):
        x1 = self.in_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out_conv(x)

        return x

if __name__ == '__main__':

    x1 = torch.randn([2, 1, 160, 160])
    model = network(1, 4)
    out = model(x1)
    print(out.shape)
    print(out)


