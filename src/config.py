# 模型参数
BASE_CONV = 64  # 隐藏网络基础通道数
BASE_REV_CONV = 8  # 提取网络基础通道数
OUT_CHANNELS = 3  # 输出通道数

# 训练参数
HIDE_WEIGHT_PATH = './params/hide.pth'  # 隐藏网络权重路径
REVEAL_WEIGHT_PATH = './params/reveal.pth'  # 提取网络权重路径
BATCH_SIZE = 8  # Batch大小
EPOCH_NUM = 300  # 训练轮次
ONE_TRAIN_QUANTITY = 200
USE_DILATION = False  # 是否使用膨胀卷积

# 其他参数
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
IMG_PATH = './img'  # 图片路径
IMG_SIZE = 256  # 图片规格

# 窗口相关参数
WIDTH = 960
HEIGHT = 600
ICON_PATH = 'pic/icon.png'
BACKGROUND_PATH_HIDE = 'pic/background_hide.png'
BACKGROUND_PATH_REVEAL = 'pic/background_reveal.png'
