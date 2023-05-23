import os.path

import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.Model import *
from src.SSIM import SSIM
from src.Utils import *


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(MyDataset(IMG_PATH, train=True), batch_size=BATCH_SIZE, shuffle=True)

    # 生成模型实例
    hide_net = Hide()
    reveal_net = Reveal()

    # 导入损失函数
    ssim = SSIM()

    # 加载权重
    if os.path.exists(HIDE_WEIGHT_PATH):
        hide_net.load_state_dict(torch.load(HIDE_WEIGHT_PATH))
        print("successfully load hide weight")
    else:
        print("not successfully load hide weight")
    if os.path.exists(REVEAL_WEIGHT_PATH):
        reveal_net.load_state_dict(torch.load(REVEAL_WEIGHT_PATH))
        print("successfully load reveal weight")
    else:
        print("not successfully load reveal weight")

    # 将网络放到设备上进行计算
    hide_net.to(device)
    reveal_net.to(device)
    ssim.to(device)

    # 优化器
    optim_hide = optim.Adam(hide_net.parameters(), lr=0.0001)
    optim_reveal = optim.Adam(reveal_net.parameters(), lr=0.0001)

    # 学习速率调整:初试学习速率为0.05；epoch为30时变为0.001；epoch为100时变为0.0002...以此类推
    # schedule_hide = MultiStepLR(optim_hide, milestones=[30, 50, 100, 150], gamma=0.2)
    # schedule_reveal = MultiStepLR(optim_reveal, milestones=[30, 50, 100, 150], gamma=0.2)

    # 进行300次训练,每个batch有BATCH_SIZE个样本；shuffle=True：每次乱序读取
    losses_hide = []
    losses_reveal = []
    for epoch in range(EPOCH_NUM + 1):

        # 初始化损失值
        epoch_loss_hide = 0.
        epoch_loss_reveal = 0.

        # 进行一轮训练
        losses = []
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        for index, (carrier_img, secret_img) in loop:
            # 将数据放到设备上
            carrier_img = carrier_img.to(device)
            secret_img = secret_img.to(device)

            # 将梯度初始化为零
            # 张量梯度是不会清零的。在每一次反向传播采用autograd计算梯度的时候，是累加的。
            # 所以应当在梯度求导之前（backward之前）把梯度清零。
            optim_hide.zero_grad()
            optim_reveal.zero_grad()

            # 利用隐藏网络进行信息隐藏，并计算隐藏损失
            hiding_carrier = hide_net(secret_img, carrier_img)

            loss_hide = 1 - ssim(carrier_img, hiding_carrier)

            # 提取秘密图像，并计算提取损失
            reveal_secret = reveal_net(hiding_carrier)
            loss_reveal = 1 - ssim(secret_img, reveal_secret)

            epoch_loss_hide += loss_hide.item()
            epoch_loss_reveal += loss_reveal.item()

            # 计算整体损失
            loss = loss_hide + 0.75 * loss_reveal
            losses.append(loss)

            # 输出损失
            # print(loss)
            loop.set_description(f'Epoch [{epoch + 1}/{EPOCH_NUM}]')
            loop.set_postfix(loss=(epoch_loss_hide, epoch_loss_reveal))

            # 自动求导函数
            loss.backward()

            # 更新网络模型
            optim_hide.step()
            optim_reveal.step()

            # 将此次模型中的秘密图片、解密后的图片、载体图片以及加密图片进行输出
            if index == 3 and (epoch + 1) % 20 == 0:  # 每20轮检测一次
                save_image(torch.cat(
                    [secret_img.cpu().data[:4], reveal_secret.cpu().data[:4], carrier_img.cpu().data[:4],
                     hiding_carrier.cpu().data[:4]],
                    dim=0), fp='./result/res_epoch_{}.png'.format(epoch + 1), nrow=4)

        losses_hide.append(epoch_loss_hide)
        losses_reveal.append(epoch_loss_reveal)

        # 更新优化器的学习率
        # schedule_hide.step()
        # schedule_reveal.step()

        # 每50轮保存一次权重
        if (epoch + 1) % 50 == 0:  # 保存模型参数，不保存结构
            torch.save(hide_net.state_dict(), './params/epoch_{}_hide.pkl'.format(epoch + 1))
            torch.save(reveal_net.state_dict(), './params/epoch_{}_reveal.pkl'.format(epoch + 1))

    # 显示损失变换
    plt.plot(losses_hide)
    plt.plot(losses_reveal)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("./result/pic.png")
    plt.show()


if __name__ == '__main__':
    main()
