import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import *


# 双层膨胀卷积
class DoubleDilaConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=USE_DILATION, mid_channels=None):  # 默认不使用膨胀卷积
        if mid_channels is None:
            mid_channels = out_channels
        if dilation:
            pad = 2
            dil = 2
        else:
            pad = 1
            dil = 1
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=pad, dilation=dil, bias=True),
            nn.LeakyReLU(inplace=True)
        )


# 下采样
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleDilaConv(in_channels, out_channels)
        )


# 上采样
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv2 = DoubleDilaConv(in_channels, out_channels)
        self.conv3 = DoubleDilaConv((in_channels // 2) * 3, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor = None):
        x1 = self.up(x1)  # 待上采样的张量
        # 进行拼接 张量参数：[N, C, H, W]，分别为批次大小N，通道数C，高H，宽W
        if x3 is not None:
            x = torch.cat([x1, x2, x3], dim=1)
            x = self.conv3(x)
        else:
            x = torch.cat([x1, x2], dim=1)
            x = self.conv2(x)
        return x


# 隐藏图像网络
class Hide(nn.Module):
    def __init__(self, base_conv=BASE_CONV):
        super().__init__()
        # 秘密图像
        self.sec_in_conv = DoubleDilaConv(in_channels=3, out_channels=base_conv)
        self.sec_down1 = Down(base_conv, base_conv * 2)
        self.sec_down2 = Down(base_conv * 2, base_conv * 4)
        self.sec_down3 = Down(base_conv * 4, base_conv * 8)
        self.sec_up1 = Up(base_conv * 8, base_conv * 4)
        self.sec_up2 = Up(base_conv * 4, base_conv * 2)
        self.sec_up3 = Up(base_conv * 2, base_conv)
        # 载体图像
        self.car_in_conv = DoubleDilaConv(in_channels=3, out_channels=base_conv)
        self.car_down1 = Down(base_conv, base_conv * 2)
        self.car_down2 = Down(base_conv * 2, base_conv * 4)
        self.car_down3 = Down(base_conv * 4, base_conv * 8)
        # 对于载体图像上采样，除本身上采样的特征图外，还包括了对应下采样的特征图和秘密信息图像预处理网络上采样过程的特征图
        self.car_up1 = Up(base_conv * 8, base_conv * 4)
        self.car_up2 = Up(base_conv * 4, base_conv * 2)
        self.car_up3 = Up(base_conv * 2, base_conv)
        self.car_out = DoubleDilaConv(in_channels=base_conv, out_channels=OUT_CHANNELS)
        # 将输出限制在[0,1]之间
        self.out = nn.Sigmoid()

    def forward(self, secret_image: torch.Tensor, carrier_image: torch.Tensor):
        sec_x1 = self.sec_in_conv(secret_image)
        sec_x2 = self.sec_down1(sec_x1)
        sec_x3 = self.sec_down2(sec_x2)
        sec_x4 = self.sec_down3(sec_x3)
        sec_up_x3 = self.sec_up1(sec_x4, sec_x3)
        sec_up_x2 = self.sec_up2(sec_up_x3, sec_x2)
        sec_up_x1 = self.sec_up3(sec_up_x2, sec_x1)

        car_x1 = self.car_in_conv(carrier_image)
        car_x2 = self.car_down1(car_x1)
        car_x3 = self.car_down2(car_x2)
        car_x4 = self.car_down3(car_x3)
        car = self.car_up1(car_x4, car_x3, sec_up_x3)
        car = self.car_up2(car, car_x2, sec_up_x2)
        car = self.car_up3(car, car_x1, sec_up_x1)
        car = self.car_out(car)
        img = self.out(car)
        return img


# 残差网络模块
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=USE_DILATION, leaky_relu=True):
        super().__init__()
        if dilation:
            pad = 2
            dil = 2
        else:
            pad = 1
            dil = 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leaky_relu = leaky_relu

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=pad, dilation=dil, bias=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        res_x = self.conv1(x)
        res_x = self.leaky_relu(res_x)
        res_x = self.conv2(res_x)
        if self.in_channels == self.out_channels:
            identity = x
        else:
            identity = self.conv(x)

        return res_x + identity


# 提取密图像网络
class Reveal(nn.Module):
    def __init__(self):
        super(Reveal, self).__init__()

        # 解码器，重建密图
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 10, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(3, 5, kernel_size=5, padding=2)

        self.conv4 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(65, 5, kernel_size=5, padding=2)

        self.conv7 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv9 = nn.Conv2d(65, 5, kernel_size=5, padding=2)

        self.conv10 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv12 = nn.Conv2d(65, 5, kernel_size=5, padding=2)

        self.conv13 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv15 = nn.Conv2d(65, 5, kernel_size=5, padding=2)

        self.conv16 = nn.Conv2d(65, 3, kernel_size=3, padding=1)

        self.res_block = ResNetBlock(in_channels=65, out_channels=65)

        # 将输出限制在[0,1]之间
        self.out = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = torch.relu(self.conv3(x))
        x4 = torch.cat((x1, x2, x3), 1)
        x4 = self.res_block(x4)

        x1 = torch.relu(self.conv4(x4))
        x2 = torch.relu(self.conv5(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = torch.relu(self.conv6(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        x4 = self.res_block(x4)

        x1 = torch.relu(self.conv7(x4))
        x2 = torch.relu(self.conv8(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = torch.relu(self.conv9(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        x4 = self.res_block(x4)

        x1 = torch.relu(self.conv10(x4))
        x2 = torch.relu(self.conv11(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = torch.relu(self.conv12(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        x4 = self.res_block(x4)

        x1 = torch.relu(self.conv13(x4))
        x2 = torch.relu(self.conv14(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = torch.relu(self.conv15(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        x4 = self.res_block(x4)

        output = self.out(self.conv16(x4))

        return output
