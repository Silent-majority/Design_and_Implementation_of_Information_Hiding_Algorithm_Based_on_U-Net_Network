import os.path
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showerror

from PIL import ImageTk

from src.Model import *
from src.Utils import *

# 设备
device = ''

# 生成模型实例
hide_net = Hide()
reveal_net = Reveal()

# 图片
secret_image = torch.Tensor()  # 秘密图像
carrier_image = torch.Tensor()  # 载体图像
res_image = torch.Tensor()  # 输出的含秘图像
carrier_secret_image = torch.Tensor()  # 用户上传的含秘图片
res_secret_image = torch.Tensor()  # 提取出的秘密图像

# 背景图片
background_image_hide = None
background_image_reveal = None


# 模型初始化
def ini():
    # 全局变量
    global device
    global hide_net
    global reveal_net
    global secret_image, carrier_image, res_image, carrier_secret_image, res_secret_image

    secret_image = carrier_image = res_image = carrier_secret_image = res_secret_image = None

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载权重
    if os.path.exists(HIDE_WEIGHT_PATH) and os.path.exists(REVEAL_WEIGHT_PATH):
        hide_net.load_state_dict(torch.load(HIDE_WEIGHT_PATH))
        reveal_net.load_state_dict(torch.load(REVEAL_WEIGHT_PATH))
    else:
        FileNotFoundError("can't load weight")

    # 将网络放到设备上进行计算
    hide_net.to(device)
    reveal_net.to(device)

    print('初始化模型。。。。。。')
    tensor = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    reveal_net(tensor)
    print('初始化完成')


# 加载文件
def load_file(img: Label, flag: str):
    global secret_image
    global carrier_image
    global carrier_secret_image
    filepath = filedialog.askopenfilename()
    if any(filepath.endswith(extension) for extension in IMG_EXTENSIONS):
        if flag == "secret":  # 加载秘密图片
            secret_image = trans(Image.open(filepath))
            image = transforms.ToPILImage()(secret_image)
        elif flag == "carrier":  # 加载载体图片
            carrier_image = trans(Image.open(filepath))
            image = transforms.ToPILImage()(carrier_image)
        else:  # 加载用户上传含秘图片
            carrier_secret_image = trans(Image.open(filepath))
            image = transforms.ToPILImage()(carrier_secret_image)
        photo = ImageTk.PhotoImage(image)
        img.configure(image=photo)
        img.image = photo
    else:
        FileNotFoundError("can't load file")


# 保存文件
def save_file(flag: str):
    if flag == "carrier_img":  # 保存含秘图片
        if res_image is None:
            showerror(title="无文件", message="请先嵌入信息")
            return
        else:
            filepath = filedialog.asksaveasfilename(title=u"保存图片", filetypes=[('PNG', 'png')])
            image = transforms.ToPILImage()(res_image)
    else:  # 保存提取出来的秘密图片
        if res_secret_image is None:
            showerror(title="无文件", message="请先提取信息")
            return
        else:
            filepath = filedialog.asksaveasfilename(title=u"保存图片", filetypes=[('PNG', 'png')])
            image = transforms.ToPILImage()(res_secret_image)
    if not filepath.endswith('.PNG'):
        filepath = str(filepath) + '.PNG'
    image.save(filepath)


# 信息嵌入
def hide(img: Label):
    global secret_image
    global carrier_image
    global res_image
    if secret_image is None:
        showerror(title="未选择文件", message="请选择秘密图像")
    elif carrier_image is None:
        showerror(title="未选择文件", message="请选择载体图像")
    else:
        # 转换格式
        secret_img = secret_image.unsqueeze(0)
        carrier_img = carrier_image.unsqueeze(0)

        # 将数据放到设备上
        secret_img = secret_img.to(device)
        carrier_img = carrier_img.to(device)

        # 获取含秘图像
        print('信息嵌入。。。')
        res_image = hide_net(secret_img, carrier_img).cpu()[0]
        print('嵌入完成')
        image = transforms.ToPILImage()(res_image)

        # 显示该图像
        photo = ImageTk.PhotoImage(image)
        img.configure(image=photo)
        img.image = photo


# 信息提取
def reveal(img: Label):
    global carrier_secret_image
    global res_secret_image

    if carrier_secret_image is None:
        showerror(title="未选择文件", message="请选择秘密图像")
    else:
        # 转换格式
        carrier_secret_img = carrier_secret_image.unsqueeze(0)

        # 将数据放到设备上
        carrier_secret_img = carrier_secret_img.to(device)

        # 提取秘密信息
        print('信息提取。。。')
        res_secret_image = reveal_net(carrier_secret_img).cpu()[0]
        print('提取完成')
        image = transforms.ToPILImage()(res_secret_image)

        # 显示该图像
        photo = ImageTk.PhotoImage(image)
        img.configure(image=photo)
        img.image = photo


# 图片隐藏界面
def get_hiding_frame(parent):
    frame = Frame(parent)

    # 设置背景图片
    global background_image_hide
    background_image_hide = ImageTk.PhotoImage(Image.open(BACKGROUND_PATH_HIDE).resize((WIDTH, HEIGHT)))
    canvas = Canvas(frame, width=WIDTH, height=HEIGHT, highlightthickness=0)
    canvas.create_image(0, 0, anchor=tkinter.NW, image=background_image_hide)
    canvas.pack()

    # 图片显示
    secret_img = Label(frame, text="秘密图像", anchor="center")
    secret_img.place(x=50, y=70, width=256, height=256)

    carrier_img = Label(frame, text="载体图像", anchor="center")
    carrier_img.place(x=350, y=70, width=256, height=256)

    hiding_carrier = Label(frame, text="含秘图像", anchor="center")
    hiding_carrier.place(x=650, y=70, width=256, height=256)

    # 文件选择和保存
    secret_button = Button(frame, text="选择文件")
    secret_button.place(x=150, y=420, width=60, height=40)

    carrier_button = Button(frame, text="选择文件")
    carrier_button.place(x=450, y=420, width=60, height=40)

    save_button = Button(frame, text="另存为")
    save_button.place(x=750, y=420, width=60, height=40)

    # 文字显示
    global sec_text_bac_hide
    global car_text_bac_hide
    global res_text_bac_hide

    sec_text_bac_hide = ImageTk.PhotoImage(
        Image.open(BACKGROUND_PATH_HIDE).resize((WIDTH, HEIGHT)).crop((120, 340, 120 + 120, 340 + 60)))
    sec_text = Label(frame, image=sec_text_bac_hide, text="秘密图像", font=('楷体', 20, 'bold'), anchor="center",
                     compound='center')
    sec_text.place(x=120, y=340, width=120, height=60)

    car_text_bac_hide = ImageTk.PhotoImage(
        Image.open(BACKGROUND_PATH_HIDE).resize((WIDTH, HEIGHT)).crop((420, 340, 420 + 120, 340 + 60)))
    car_text = Label(frame, image=car_text_bac_hide, text="载体图像", font=('楷体', 20, 'bold'), anchor="center",
                     compound='center')
    car_text.place(x=420, y=340, width=120, height=60)

    res_text_bac_hide = ImageTk.PhotoImage(
        Image.open(BACKGROUND_PATH_HIDE).resize((WIDTH, HEIGHT)).crop((720, 340, 720 + 120, 340 + 60)))
    res_text = Label(frame, image=res_text_bac_hide, text="含秘图像", font=('楷体', 20, 'bold'), anchor="center",
                     compound='center')
    res_text.place(x=720, y=340, width=120, height=60)

    # 文本声明
    txt = "山东科技大学本科毕业设计\n计算机科学与技术19级2班\n作者：李昌乐"
    global background_msg_hide
    background_msg_hide = ImageTk.PhotoImage(
        Image.open(BACKGROUND_PATH_HIDE).resize((WIDTH, HEIGHT)).crop((760, 520, 760 + 200, 520 + 80)))
    msg = Label(frame, image=background_msg_hide, text=txt, width=160, font=('微软雅黑', 8), compound='center',
                justify='center')
    msg.place(x=760, y=520, width=200, height=80)

    # 隐藏按钮
    information_hiding_button = Button(frame, text="信息嵌入")
    information_hiding_button.place(x=270, y=480, width=120, height=80)

    # 绑定事件
    secret_button.bind('<Button-1>', lambda event: load_file(secret_img, "secret"))
    carrier_button.bind('<Button-1>', lambda event: load_file(carrier_img, "carrier"))
    save_button.bind('<Button-1>', lambda event: save_file("carrier_img"))
    information_hiding_button.bind('<Button-1>', lambda event: hide(hiding_carrier))
    return frame


# 图片提取界面
def get_reveal_frame(parent):
    frame = Frame(parent)

    # 设置背景图片
    global background_image_reveal
    background_image_reveal = ImageTk.PhotoImage(Image.open(BACKGROUND_PATH_REVEAL).resize((WIDTH, HEIGHT)))
    canvas = Canvas(frame, width=WIDTH, height=HEIGHT, highlightthickness=0)
    canvas.create_image(0, 0, anchor=tkinter.NW, image=background_image_reveal)
    canvas.pack()

    # 图片显示
    carrier_img = Label(frame, text="含秘图像", anchor="center")
    carrier_img.place(x=120, y=80, width=256, height=256)

    secret_img = Label(frame, text="秘密图像", anchor="center")
    secret_img.place(x=580, y=80, width=256, height=256)

    # 文件选择和保存
    carrier_button = Button(frame, text="选择文件")
    carrier_button.place(x=220, y=440, width=60, height=40)

    save_button = Button(frame, text="另存为")
    save_button.place(x=680, y=440, width=60, height=40)

    # 文字显示
    global sec_text_bac_reveal
    global car_text_bac_reveal

    car_text_bac_reveal = ImageTk.PhotoImage(
        Image.open(BACKGROUND_PATH_REVEAL).resize((WIDTH, HEIGHT)).crop((190, 360, 190 + 120, 360 + 60)))
    car_text = Label(frame, image=car_text_bac_reveal, text="含秘图像", font=('楷体', 20, 'bold'), anchor="center",
                     compound="center")
    car_text.place(x=190, y=360, width=120, height=60)

    sec_text_bac_reveal = ImageTk.PhotoImage(
        Image.open(BACKGROUND_PATH_REVEAL).resize((WIDTH, HEIGHT)).crop((650, 360, 650 + 120, 360 + 60)))
    sec_text = Label(frame, image=sec_text_bac_reveal, text="秘密图像", font=('楷体', 20, 'bold'), anchor="center",
                     compound="center")
    sec_text.place(x=650, y=360, width=120, height=60)

    # 文本声明
    txt = "山东科技大学本科毕业设计\n计算机科学与技术19级2班\n作者：李昌乐"
    global background_msg_reveal
    background_msg_reveal = ImageTk.PhotoImage(
        Image.open(BACKGROUND_PATH_REVEAL).resize((WIDTH, HEIGHT)).crop((760, 520, 760 + 200, 520 + 80)))
    msg = Label(frame, image=background_msg_reveal, text=txt, width=160, font=('微软雅黑', 8), compound='center',
                justify='center')
    msg.place(x=760, y=520, width=200, height=80)

    # 隐藏按钮
    information_reveal_button = Button(frame, text="信息提取")
    information_reveal_button.place(x=420, y=460, width=120, height=80)

    # 绑定事件
    carrier_button.bind('<Button-1>', lambda event: load_file(carrier_img, "carrier_secret"))
    save_button.bind('<Button-1>', lambda event: save_file("reveal_img"))
    information_reveal_button.bind('<Button-1>', lambda event: reveal(secret_img))
    return frame


class MainScene(Tk):
    def __init__(self):
        super().__init__()
        self.title('秘密图像嵌入与提取')
        self.resizable(width=False, height=False)

        # 窗口居中，获取屏幕尺寸以计算布局参数，使窗口居屏幕中央
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        self.geometry(
            '{}x{}+{}+{}'.format(WIDTH, HEIGHT, (screenwidth - WIDTH) // 2, (screenheight - HEIGHT) // 2))

        # 获取两界面
        self.hiding_frame = get_hiding_frame(self)
        self.reveal_frame = get_reveal_frame(self)

        # 当前显示隐藏界面
        self.current_page = self.hiding_frame
        self.current_page.place(relx=0., rely=0., relwidth=1., relheight=1.)

        # 菜单，用于两界面的切换
        main_menu = Menu(self)
        main_menu.add_command(label="信息隐藏", command=self.hiding_page)
        main_menu.add_command(label="信息提取", command=self.reveal_page)
        self.config(menu=main_menu)

    # 切换隐藏界面
    def hiding_page(self):
        if self.current_page != self.hiding_frame:
            self.current_page.place_forget()
            self.current_page = self.hiding_frame
            self.current_page.place(relx=0., rely=0., relwidth=1., relheight=1.)

    # 切换提取界面
    def reveal_page(self):
        if self.current_page != self.reveal_frame:
            self.current_page.place_forget()
            self.current_page = self.reveal_frame
            self.current_page.place(relx=0., rely=0., relwidth=1., relheight=1.)


def main():
    ini()
    main_scene = MainScene()
    main_scene.mainloop()


if __name__ == '__main__':
    main()
