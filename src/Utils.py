import math
import os.path

import numpy
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import *


def trans(image: Image):
    image = transforms.Resize([IMG_SIZE, IMG_SIZE])(image).convert('RGB')
    image = transforms.ToTensor()(image)
    if image.shape[0] != 3:
        image = torch.cat([image, image, image], dim=0)
    return image


class MyDataset(Dataset):
    def __init__(self, path: str, train: bool):
        super().__init__()

        self.flag = "train" if train else "test"
        img_path = os.path.join(path, self.flag)
        assert os.path.exists(img_path), f"文件路径 '{img_path}' 不存在."

        # 获取载体图片和隐藏图片
        self.carrier_filenames = []
        self.secret_filenames = []
        #
        # start = ONE_TRAIN_QUANTITY * 5
        # start = 0
        for i, filename in enumerate(os.listdir(img_path)[1200:1800]):
            if i % 2 == 0:
                self.carrier_filenames.append(os.path.join(img_path, filename))
            else:
                self.secret_filenames.append(os.path.join(img_path, filename))

        assert len(self.carrier_filenames) == len(self.secret_filenames), f"数量不匹配"

        # 检查文件
        for i in self.carrier_filenames:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        for i in self.secret_filenames:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    # 根据索引index遍历数据
    def __getitem__(self, idx):
        carrier_img = Image.open(self.carrier_filenames[idx])
        secret_img = Image.open(self.secret_filenames[idx])

        # 作为图像的预处理，设定图像的大小的裁剪形式
        carrier_img = trans(carrier_img)
        secret_img = trans(secret_img)

        if carrier_img.shape[0] != 3 or secret_img.shape[0] != 3:
            raise RuntimeError(
                "shape error {0} and {1} or {2} and {3}".format(carrier_img.shape[0], self.carrier_filenames[idx],
                                                                secret_img.shape[0], self.secret_filenames[idx]))
        return carrier_img, secret_img

    # 返回数据集的长度
    def __len__(self):
        return len(self.secret_filenames)


def PSNR(img1, img2):
    mse = numpy.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)
